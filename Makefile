CC = gcc
CFLAGS =
LDFLAGS = -lm

objects = ann.o mt19937-64.o

build : $(objects)
	$(CC) -o ann.out $(CFLAGS) $(LDFLAGS) $(objects)

ann.o : ann.h mt19937-64.h
	$(CC) -c ann.c

mt19937-64.o :
	$(CC) -c mt19937-64.c

clean:
	rm ann.out $(objects)
