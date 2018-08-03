CC = gcc
CFLAGS = -Wall
LDFLAGS = -lgsl -lcblas

objects = ann.o

build : $(objects)
	$(CC) $(CFLAGS) $(LDFLAGS) -o ann.out $(objects)

ann.o :
	$(CC) -c ann.c

clean:
	rm ann.out $(objects)
