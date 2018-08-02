CC=gcc
CFLAGS=-I/usr/lib/modules/$(uname -r)/build/include/linux
LDFLAGS=-lm

build:
	$(CC) $(CFLAGS) $(LDFLAGS) ann.c -o ann.out

clean:
	rm ann.out
