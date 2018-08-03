INCL =
SRC = ann.c
OBJ = $(SRC:.c=.o)
LIBS = -lgsl -lcblas
EXE = ann.out

CC = gcc
CFLAGS = -Wall -Og -g
LIBPATH = -L.
LDFLAGS = -o $(EXE) $(LIBPATH) $(LIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

$(EXE): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ)

$(OBJ): $(INCL)

clean:
	rm $(OBJ) $(EXE)
