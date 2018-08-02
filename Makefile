CC=gcc
CFLAGS=
LDFLAGS=-lm

build: ann.h
	$(CC) $(CFLAGS) $(LDFLAGS) ann.c -o ann.out

clean:
	rm ann.out
