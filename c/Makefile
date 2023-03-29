CC=gcc
CFLAGS=-Wextra -Wall -pedantic -std=c99 -g
LDFLAGS=-lm

all: test

spai: spai.c
	$(CC) -o spai spai.c $(CFLAGS)

csc.o: csc.c csc.h
	$(CC) -c $< -o $@

test: test.o csc.o
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) -c $< $(CFLAGS)

clean:
	rm -rf spai