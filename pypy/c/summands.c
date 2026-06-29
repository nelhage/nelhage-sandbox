#include <stdlib.h>

#include "summands.h"

/* Each concrete type has summand_base as its first member, so a summand_base*
   can be cast to the concrete type inside the function. */

typedef struct {
    summand_base base;
    int value;
} add_int;

typedef struct {
    summand_base base;
    int value;
} square_int;

typedef struct {
    summand_base base;
    int value; /* unused, but keeps the layout parallel */
} identity;

static int add_int_fn(summand_base *b, int running) {
    add_int *s = (add_int *)b;
    return running + s->value;
}

static int square_int_fn(summand_base *b, int running) {
    square_int *s = (square_int *)b;
    return running + s->value * s->value;
}

static int identity_fn(summand_base *b, int running) {
    (void)b;
    return running;
}

summand_base *make_add_int(int value) {
    add_int *s = malloc(sizeof *s);
    s->base.func = add_int_fn;
    s->value = value;
    return &s->base;
}

summand_base *make_square_int(int value) {
    square_int *s = malloc(sizeof *s);
    s->base.func = square_int_fn;
    s->value = value;
    return &s->base;
}

summand_base *make_identity(int value) {
    identity *s = malloc(sizeof *s);
    s->base.func = identity_fn;
    s->value = value;
    return &s->base;
}

summand_ctor *const SUMMAND_KINDS[3] = {
    make_add_int,
    make_square_int,
    make_identity,
};

int polysum(summand_base **summands, size_t n) {
    int total = 0;
    for (size_t i = 0; i < n; i++) {
        summand_base *b = summands[i];
        total = b->func(b, total);
    }
    return total;
}
