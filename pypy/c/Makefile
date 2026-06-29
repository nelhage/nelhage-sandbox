# Idiomatic optimized build. -O2 keeps strict IEEE FP semantics (so the plain
# float sum stays a scalar reduction, matching the other engines); add
# -ffast-math -march=native to let the compiler vectorize it.
CFLAGS ?= -O2 -Wall -Wextra -std=gnu11
LDLIBS = -lm

all: benchmark sweep_nansum

benchmark: benchmark.c sum_floats.c summands.c sum_floats.h summands.h bench_util.h
	$(CC) $(CFLAGS) -o $@ benchmark.c sum_floats.c summands.c $(LDLIBS)

sweep_nansum: sweep_nansum.c sum_floats.c sum_floats.h bench_util.h
	$(CC) $(CFLAGS) -o $@ sweep_nansum.c sum_floats.c $(LDLIBS)

clean:
	rm -f benchmark sweep_nansum

.PHONY: all clean
