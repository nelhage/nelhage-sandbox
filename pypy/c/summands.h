#ifndef SUMMANDS_H
#define SUMMANDS_H

#include <stddef.h>

/* Polymorphic dispatch via a one-entry vtable: every summand starts with a
   summand_base holding a function pointer. Concrete types embed summand_base
   as their first member and recover their full type with a cast inside their
   summand_fn. polysum walks an array of summand_base* and calls through func,
   which the CPU resolves as an indirect branch. */
typedef struct summand_base summand_base;

typedef int summand_fn(summand_base *, int running);

struct summand_base {
    summand_fn *func;
};

/* Constructors: heap-allocate a concrete summand and return its base. */
summand_base *make_add_int(int value);
summand_base *make_square_int(int value);
summand_base *make_identity(int value);

/* Constructor signature, for building a random mix. */
typedef summand_base *summand_ctor(int value);
extern summand_ctor *const SUMMAND_KINDS[3];

int polysum(summand_base **summands, size_t n);

#endif /* SUMMANDS_H */
