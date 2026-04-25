/*
 * ariannamethod.c — AML mini-compiler for penelope
 * Subset of the AriannaMethod language compiler.
 * Transpiles .aml → C → executable.
 *
 * Features: BLOOD COMPILE, BLOOD MAIN, BLOOD LINK, ECHO, comments
 *
 * Part of the AriannaMethod project (github.com/ariannamethod/ariannamethod.ai)
 * By Arianna Method. הרזוננס לא נשבר
 *
 * Build:
 *   cc ariannamethod.c -o amlc
 *
 * Usage:
 *   ./amlc penelope.aml              # compile to ./penelope_aml
 *   ./amlc penelope.aml -o penelope  # compile to ./penelope
 *   ./amlc penelope.aml --emit-c     # emit generated C to stdout
 *   ./amlc penelope.aml --run        # compile and run
 *   ./amlc penelope.aml --run -- arg1 arg2  # compile, run with args
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/wait.h>

/* ── limits ────────────────────────────────────────────────────────── */

#define MAX_BLOCKS      256
#define MAX_ECHOS       1024
#define INIT_BUF        (1 << 20)   /* 1 MiB initial code buffer */
#define GROW_FACTOR     2

/* ── data structures ───────────────────────────────────────────────── */

/* A named BLOOD COMPILE block: verbatim C code */
typedef struct {
    char name[256];
    char *code;         /* heap-allocated */
    size_t len;
} BloodBlock;

/* An ECHO statement: text to print at runtime */
typedef struct {
    char text[1024];
} EchoStmt;

/* Parsed AML program */
typedef struct {
    BloodBlock blocks[MAX_BLOCKS];
    int        nblocks;

    char       links[MAX_BLOCKS][256]; /* BLOOD LINK names */
    int        nlinks;

    EchoStmt   echos[MAX_ECHOS];
    int        nechos;

    char      *main_code;   /* BLOOD MAIN body, or NULL */
    size_t     main_len;
} AmlProgram;

/* ── dynamic buffer ────────────────────────────────────────────────── */

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} Buf;

static void buf_init(Buf *b)
{
    b->cap  = INIT_BUF;
    b->data = malloc(b->cap);
    if (!b->data) { perror("malloc"); exit(1); }
    b->len  = 0;
    b->data[0] = '\0';
}

static void buf_append(Buf *b, const char *s, size_t n)
{
    while (b->len + n + 1 > b->cap) {
        b->cap *= GROW_FACTOR;
        b->data = realloc(b->data, b->cap);
        if (!b->data) { perror("realloc"); exit(1); }
    }
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static void buf_printf(Buf *b, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void buf_printf(Buf *b, const char *fmt, ...)
{
    va_list ap, ap2;
    char tmp[4096];

    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);

    if (n < 0) { va_end(ap2); return; }

    if ((size_t)n < sizeof(tmp)) {
        va_end(ap2);
        buf_append(b, tmp, (size_t)n);
    } else {
        /* large format: heap-alloc */
        char *big = malloc((size_t)n + 1);
        if (!big) { perror("malloc"); va_end(ap2); exit(1); }
        vsnprintf(big, (size_t)n + 1, fmt, ap2);
        va_end(ap2);
        buf_append(b, big, (size_t)n);
        free(big);
    }
}

static void buf_free(Buf *b)
{
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

/* ── helpers ───────────────────────────────────────────────────────── */

/* Read entire file into heap-allocated string. Returns NULL on error. */
static char *read_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);

    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    if (out_len) *out_len = rd;
    return buf;
}

/* Strip leading whitespace in-place (returns pointer into same string) */
static const char *skip_ws(const char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

/* Check if string starts with prefix (case-sensitive) */
static int starts_with(const char *s, const char *prefix)
{
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

/* Count newlines in a string */
static int count_lines(const char *s)
{
    int n = 0;
    for (; *s; s++) if (*s == '\n') n++;
    return n;
}

/* ── parser ────────────────────────────────────────────────────────── */

/*
 * Collect a brace-delimited block of C code.
 * 'p' points to the first '{'.
 * Returns pointer past the closing '}'.
 * Writes the contents (between braces) into dst.
 */
static const char *collect_braced(const char *p, Buf *dst)
{
    if (*p != '{') return p;
    p++; /* skip opening brace */

    int depth = 1;
    while (*p && depth > 0) {
        if (*p == '{') depth++;
        else if (*p == '}') { depth--; if (depth == 0) break; }
        buf_append(dst, p, 1);
        p++;
    }
    if (*p == '}') p++; /* skip closing brace */
    return p;
}

/*
 * Parse an .aml source string into an AmlProgram.
 * Returns 0 on success, -1 on error.
 */
static int parse_aml(const char *src, AmlProgram *prog)
{
    memset(prog, 0, sizeof(*prog));

    const char *p = src;
    int lineno = 0;

    while (*p) {
        /* extract one line */
        const char *eol = strchr(p, '\n');
        if (!eol) eol = p + strlen(p);
        size_t line_len = (size_t)(eol - p);

        char line[4096];
        if (line_len >= sizeof(line)) line_len = sizeof(line) - 1;
        memcpy(line, p, line_len);
        line[line_len] = '\0';
        lineno++;

        const char *trimmed = skip_ws(line);

        /* skip empty lines and comments */
        if (*trimmed == '\0' || *trimmed == '#') {
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        /* ── BLOOD COMPILE name { ... } ──────────────────────────── */
        if (starts_with(trimmed, "BLOOD COMPILE ")) {
            if (prog->nblocks >= MAX_BLOCKS) {
                fprintf(stderr, "amlc: error: too many BLOOD COMPILE blocks (max %d)\n", MAX_BLOCKS);
                return -1;
            }

            const char *rest = trimmed + strlen("BLOOD COMPILE ");
            rest = skip_ws(rest);

            /* extract block name (until whitespace or '{') */
            BloodBlock *blk = &prog->blocks[prog->nblocks];
            int ni = 0;
            while (*rest && !isspace((unsigned char)*rest) && *rest != '{' && ni < 255) {
                blk->name[ni++] = *rest++;
            }
            blk->name[ni] = '\0';

            /* find the opening brace — may be on this line or the next */
            while (*rest && isspace((unsigned char)*rest)) rest++;

            /* build full-source pointer to where we are */
            const char *brace_start = p + (rest - line);

            if (*rest == '{') {
                /* brace is on this line */
                Buf body;
                buf_init(&body);
                const char *after = collect_braced(brace_start, &body);
                blk->code = body.data;
                blk->len  = body.len;
                prog->nblocks++;
                p = after;
                /* skip trailing whitespace/newline */
                while (*p && (*p == ' ' || *p == '\t')) p++;
                if (*p == '\n') p++;
                continue;
            } else {
                /* brace on next line — advance past this line */
                p = (*eol) ? eol + 1 : eol;
                /* skip blank lines until we hit '{' */
                while (*p) {
                    while (*p && isspace((unsigned char)*p) && *p != '\n') p++;
                    if (*p == '{') break;
                    if (*p == '\n') { p++; lineno++; continue; }
                    fprintf(stderr, "amlc: line %d: expected '{' for BLOOD COMPILE %s\n",
                            lineno, blk->name);
                    return -1;
                }
                if (*p == '{') {
                    Buf body;
                    buf_init(&body);
                    const char *after = collect_braced(p, &body);
                    blk->code = body.data;
                    blk->len  = body.len;
                    prog->nblocks++;
                    p = after;
                    while (*p && (*p == ' ' || *p == '\t')) p++;
                    if (*p == '\n') p++;
                    continue;
                }
            }
        }

        /* ── BLOOD MAIN { ... } ──────────────────────────────────── */
        if (starts_with(trimmed, "BLOOD MAIN")) {
            const char *rest = trimmed + strlen("BLOOD MAIN");
            while (*rest && isspace((unsigned char)*rest)) rest++;

            const char *brace_start = p + (rest - line);

            if (*rest == '{') {
                Buf body;
                buf_init(&body);
                const char *after = collect_braced(brace_start, &body);
                prog->main_code = body.data;
                prog->main_len  = body.len;
                p = after;
                while (*p && (*p == ' ' || *p == '\t')) p++;
                if (*p == '\n') p++;
                continue;
            } else {
                p = (*eol) ? eol + 1 : eol;
                while (*p) {
                    while (*p && isspace((unsigned char)*p) && *p != '\n') p++;
                    if (*p == '{') break;
                    if (*p == '\n') { p++; lineno++; continue; }
                    fprintf(stderr, "amlc: line %d: expected '{' for BLOOD MAIN\n", lineno);
                    return -1;
                }
                if (*p == '{') {
                    Buf body;
                    buf_init(&body);
                    const char *after = collect_braced(p, &body);
                    prog->main_code = body.data;
                    prog->main_len  = body.len;
                    p = after;
                    while (*p && (*p == ' ' || *p == '\t')) p++;
                    if (*p == '\n') p++;
                    continue;
                }
            }
        }

        /* ── BLOOD LINK name ─────────────────────────────────────── */
        if (starts_with(trimmed, "BLOOD LINK ")) {
            if (prog->nlinks >= MAX_BLOCKS) {
                fprintf(stderr, "amlc: error: too many BLOOD LINK directives\n");
                return -1;
            }
            const char *name = skip_ws(trimmed + strlen("BLOOD LINK "));
            char tmp[256];
            int ni = 0;
            while (*name && !isspace((unsigned char)*name) && ni < 255)
                tmp[ni++] = *name++;
            tmp[ni] = '\0';
            strncpy(prog->links[prog->nlinks], tmp, 255);
            prog->links[prog->nlinks][255] = '\0';
            prog->nlinks++;
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        /* ── ECHO text ───────────────────────────────────────────── */
        if (starts_with(trimmed, "ECHO ")) {
            if (prog->nechos >= MAX_ECHOS) {
                fprintf(stderr, "amlc: error: too many ECHO statements\n");
                return -1;
            }
            const char *text = trimmed + strlen("ECHO ");
            strncpy(prog->echos[prog->nechos].text, text, 1023);
            prog->echos[prog->nechos].text[1023] = '\0';
            prog->nechos++;
            p = (*eol) ? eol + 1 : eol;
            continue;
        }

        /* ── unknown command — warn and skip ─────────────────────── */
        /* Silently skip known AML keywords that we don't implement */
        if (starts_with(trimmed, "PROPHECY") ||
            starts_with(trimmed, "DESTINY")  ||
            starts_with(trimmed, "VELOCITY") ||
            starts_with(trimmed, "FIELD")    ||
            starts_with(trimmed, "RESONANCE")||
            starts_with(trimmed, "STEP")     ||
            starts_with(trimmed, "TRAIN")    ||
            starts_with(trimmed, "LOAD")     ||
            starts_with(trimmed, "SAVE"))
        {
            fprintf(stderr, "amlc: line %d: skipping unimplemented AML command: %.40s\n",
                    lineno, trimmed);
        } else {
            fprintf(stderr, "amlc: line %d: unknown directive: %.60s\n", lineno, trimmed);
        }

        p = (*eol) ? eol + 1 : eol;
    }

    return 0;
}

/* ── code generation ───────────────────────────────────────────────── */

/*
 * Generate C source from a parsed AML program.
 * Returns a heap-allocated Buf with the generated code.
 */
static void generate_c(const AmlProgram *prog, Buf *out)
{
    buf_init(out);

    /* header comment */
    buf_printf(out,
        "/* Generated by amlc — AriannaMethod mini-compiler */\n"
        "/* Do not edit. Regenerate from the .aml source.    */\n\n");

    /* standard includes (skip if BLOOD blocks provide their own) */
    if (prog->nblocks == 0) {
        buf_printf(out,
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "#include <string.h>\n"
            "#include <math.h>\n"
            "#include <time.h>\n"
            "#include <ctype.h>\n\n");
    }

    /* emit all BLOOD COMPILE blocks in order */
    for (int i = 0; i < prog->nblocks; i++) {
        buf_printf(out,
            "/* ── BLOOD COMPILE: %s ── */\n", prog->blocks[i].name);
        buf_append(out, prog->blocks[i].code, prog->blocks[i].len);
        buf_printf(out, "\n\n");
    }

    /* emit main() */
    if (prog->main_code) {
        /* BLOOD MAIN provided — emit verbatim (includes its own main signature) */
        buf_append(out, prog->main_code, prog->main_len);
        buf_printf(out, "\n");
    } else {
        /* auto-generate main from ECHO statements */
        buf_printf(out, "int main(int argc, char **argv)\n{\n");
        buf_printf(out, "    (void)argc; (void)argv;\n");
        for (int i = 0; i < prog->nechos; i++) {
            /* escape quotes in the echo text for C string */
            buf_printf(out, "    printf(\"  ");
            const char *t = prog->echos[i].text;
            while (*t) {
                if (*t == '"')       buf_append(out, "\\\"", 2);
                else if (*t == '\\') buf_append(out, "\\\\", 2);
                else if (*t == '%')  buf_append(out, "%%", 2);
                else                 buf_append(out, t, 1);
                t++;
            }
            buf_printf(out, "\\n\");\n");
        }
        buf_printf(out, "    return 0;\n}\n");
    }
}

/* ── compilation and execution ─────────────────────────────────────── */

/*
 * Derive a default output name from the .aml filename.
 * "penelope.aml" → "penelope_aml"
 */
static void default_output_name(const char *aml_path, char *out, size_t outsz)
{
    /* find basename */
    const char *base = strrchr(aml_path, '/');
    base = base ? base + 1 : aml_path;

    strncpy(out, base, outsz - 1);
    out[outsz - 1] = '\0';

    /* replace '.' with '_' */
    for (char *p = out; *p; p++) {
        if (*p == '.') *p = '_';
    }
}

/*
 * Write generated C to a temp file, compile with cc, return 0 on success.
 * Sets tmp_path to the temp file path (caller should unlink on success).
 */
static int compile_c(const char *c_code, size_t c_len,
                     const char *output, const char *input_dir,
                     char *tmp_path, size_t tmp_sz)
{
    /* create temp file */
    snprintf(tmp_path, tmp_sz, "/tmp/amlc_XXXXXX.c");
    /* mkstemps needs suffix length */
    int fd = mkstemps(tmp_path, 2); /* ".c" = 2 chars */
    if (fd < 0) {
        perror("amlc: mkstemps");
        return -1;
    }

    /* write generated C */
    size_t written = 0;
    while (written < c_len) {
        ssize_t w = write(fd, c_code + written, c_len - written);
        if (w <= 0) { perror("amlc: write"); close(fd); return -1; }
        written += (size_t)w;
    }
    close(fd);

    /* build cc command — sanitize paths to prevent injection */
    char cmd[4096];
    /* validate output and tmp_path contain no shell metacharacters */
    for (const char *p = output; *p; p++) {
        if (*p == '\'' || *p == '\\' || *p == '$' || *p == '`'
            || *p == '(' || *p == ')' || *p == ';' || *p == '&'
            || *p == '|' || *p == '\n') {
            fprintf(stderr, "amlc: unsafe character in output path\n");
            return -1;
        }
    }
    /* allow #include "..." inside BLOOD blocks to resolve relative
       to the AML source directory (e.g. tools/bpe_merges.h)         */
    const char *idir = (input_dir && *input_dir) ? input_dir : ".";
    for (const char *p = idir; *p; p++) {
        if (*p == '\'' || *p == '\\' || *p == '`' || *p == '$'
            || *p == '\n' || *p == ';' || *p == '|') {
            fprintf(stderr, "amlc: unsafe character in input dir\n");
            return -1;
        }
    }
    snprintf(cmd, sizeof(cmd),
             "cc -O2 -I '%s' -o '%s' '%s' -lm 2>&1",
             idir, output, tmp_path);

    fprintf(stderr, "amlc: compiling → %s\n", output);

    FILE *proc = popen(cmd, "r");
    if (!proc) {
        perror("amlc: popen cc");
        return -1;
    }

    /* capture compiler output */
    char line[1024];
    int had_output = 0;
    while (fgets(line, sizeof(line), proc)) {
        if (!had_output) {
            fprintf(stderr, "amlc: cc output:\n");
            had_output = 1;
        }
        fprintf(stderr, "  %s", line);
    }

    int status = pclose(proc);
    if (status != 0) {
        fprintf(stderr, "amlc: compilation failed (cc exit %d)\n",
                WIFEXITED(status) ? WEXITSTATUS(status) : status);
        fprintf(stderr, "amlc: temp file kept for debugging: %s\n", tmp_path);
        return -1;
    }

    return 0;
}

/*
 * Execute the compiled binary with optional arguments.
 * Does not return on success (uses execv).
 */
static int run_binary(const char *path, int run_argc, char **run_argv)
{
    /* build argv: [path, run_argv..., NULL] */
    int total = 1 + run_argc + 1;
    char **argv = malloc(sizeof(char *) * (size_t)total);
    if (!argv) { perror("malloc"); return -1; }

    argv[0] = (char *)path;
    for (int i = 0; i < run_argc; i++)
        argv[1 + i] = run_argv[i];
    argv[total - 1] = NULL;

    fprintf(stderr, "amlc: running %s", path);
    for (int i = 0; i < run_argc; i++)
        fprintf(stderr, " %s", run_argv[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "────────────────────────────────────────\n");

    execv(path, argv);
    /* if we get here, execv failed */
    perror("amlc: execv");
    free(argv);
    return -1;
}

/* ── usage ─────────────────────────────────────────────────────────── */

static void usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s <file.aml> [options]\n"
        "\n"
        "Options:\n"
        "  -o <name>     Output binary name (default: derived from input)\n"
        "  --emit-c      Print generated C to stdout (don't compile)\n"
        "  --run         Compile and run immediately\n"
        "  -- arg ...    Arguments passed to the program (with --run)\n"
        "\n"
        "Examples:\n"
        "  %s penelope.aml              # → ./penelope_aml\n"
        "  %s penelope.aml -o penelope  # → ./penelope\n"
        "  %s penelope.aml --emit-c     # print C to stdout\n"
        "  %s penelope.aml --run        # compile & run\n"
        "  %s penelope.aml --run -- \"darkness eats\"\n"
        "\n"
        "Part of the AriannaMethod project.\n",
        argv0, argv0, argv0, argv0, argv0, argv0);
}

/* ── main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    /* parse arguments */
    const char *input_file = NULL;
    const char *output_name = NULL;
    int emit_c   = 0;
    int do_run   = 0;
    int run_argc = 0;
    char **run_argv = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--emit-c") == 0) {
            emit_c = 1;
        } else if (strcmp(argv[i], "--run") == 0) {
            do_run = 1;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "amlc: -o requires an argument\n");
                return 1;
            }
            output_name = argv[++i];
        } else if (strcmp(argv[i], "--") == 0) {
            /* remaining args are for the compiled program */
            run_argc = argc - i - 1;
            run_argv = &argv[i + 1];
            break;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "amlc: unknown option: %s\n", argv[i]);
            return 1;
        } else {
            if (input_file) {
                fprintf(stderr, "amlc: multiple input files not supported\n");
                return 1;
            }
            input_file = argv[i];
        }
    }

    if (!input_file) {
        fprintf(stderr, "amlc: no input file specified\n");
        usage(argv[0]);
        return 1;
    }

    /* read source */
    size_t src_len = 0;
    char *src = read_file(input_file, &src_len);
    if (!src) {
        fprintf(stderr, "amlc: cannot read '%s': ", input_file);
        perror(NULL);
        return 1;
    }

    fprintf(stderr, "amlc: reading %s (%zu bytes)\n", input_file, src_len);

    /* parse */
    AmlProgram prog;
    if (parse_aml(src, &prog) != 0) {
        free(src);
        return 1;
    }

    fprintf(stderr, "amlc: parsed %d BLOOD block(s), %d ECHO(s), %d LINK(s)%s\n",
            prog.nblocks, prog.nechos, prog.nlinks,
            prog.main_code ? ", BLOOD MAIN present" : "");

    /* generate C */
    Buf gen;
    generate_c(&prog, &gen);

    int gen_lines = count_lines(gen.data);
    fprintf(stderr, "amlc: generated %d lines of C (%zu bytes)\n",
            gen_lines, gen.len);

    /* --emit-c: just print and exit */
    if (emit_c) {
        fwrite(gen.data, 1, gen.len, stdout);
        buf_free(&gen);
        free(src);
        return 0;
    }

    /* determine output name */
    char out_name[1024];
    if (output_name) {
        strncpy(out_name, output_name, sizeof(out_name) - 1);
        out_name[sizeof(out_name) - 1] = '\0';
    } else {
        default_output_name(input_file, out_name, sizeof(out_name));
    }

    /* derive input directory so #include "..." inside BLOOD blocks
       resolves relative to the AML source file                       */
    char input_dir[1024];
    const char *slash = strrchr(input_file, '/');
    if (slash && slash > input_file) {
        size_t dlen = (size_t)(slash - input_file);
        if (dlen >= sizeof(input_dir)) dlen = sizeof(input_dir) - 1;
        memcpy(input_dir, input_file, dlen);
        input_dir[dlen] = 0;
    } else {
        strcpy(input_dir, ".");
    }

    /* compile */
    char tmp_path[1024];
    if (compile_c(gen.data, gen.len, out_name, input_dir,
                  tmp_path, sizeof(tmp_path)) != 0) {
        buf_free(&gen);
        free(src);
        return 1;
    }

    /* clean up temp file on success */
    unlink(tmp_path);
    fprintf(stderr, "amlc: success → %s\n", out_name);

    /* optionally run */
    if (do_run) {
        /* make path absolute or relative-safe for execv */
        char run_path[2048];
        if (out_name[0] == '/') {
            strncpy(run_path, out_name, sizeof(run_path) - 1);
            run_path[sizeof(run_path) - 1] = '\0';
        } else {
            snprintf(run_path, sizeof(run_path), "./%s", out_name);
        }

        buf_free(&gen);
        free(src);

        /* does not return on success */
        run_binary(run_path, run_argc, run_argv);
        return 1; /* only reached on execv failure */
    }

    /* clean up */
    buf_free(&gen);
    free(src);
    for (int i = 0; i < prog.nblocks; i++)
        free(prog.blocks[i].code);
    if (prog.main_code)
        free(prog.main_code);

    return 0;
}
