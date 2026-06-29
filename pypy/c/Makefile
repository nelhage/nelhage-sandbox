# Idiomatic optimized build. At -O2 gcc emits no SIMD (auto-vectorization is
# -O3), but the int loops are still fast: int-add is 1-cycle latency (vs ~4 for
# float), and gcc if-converts the sentinel skip into a branchless cmov -- so
# the sentinel sweep is flat, with no branch-misprediction U-curve. Build with
# CFLAGS="-O3 ..." to let it vectorize (roughly +40-80%).
CFLAGS ?= -O2 -Wall -Wextra -std=gnu11

all: benchmark sweep_sentinel

benchmark: benchmark.c sums.c summands.c sums.h summands.h bench_util.h
	$(CC) $(CFLAGS) -o $@ benchmark.c sums.c summands.c

sweep_sentinel: sweep_sentinel.c sums.c sums.h bench_util.h
	$(CC) $(CFLAGS) -o $@ sweep_sentinel.c sums.c

clean:
	rm -f benchmark sweep_sentinel

.PHONY: all clean
