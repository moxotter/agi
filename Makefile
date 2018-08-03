CC = gcc
CFLAGS = -Wall -Og -g
LDFLAGS = -lgsl -lcblas

objects = ann.o

build : $(objects)
	$(CC) $(CFLAGS) $(LDFLAGS) -o ann.out $(objects)

ann.o :
	$(CC) $(CFLAGS) -c ann.c

clean:
	rm ann.out $(objects)
