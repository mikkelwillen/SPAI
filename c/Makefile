CC=gcc
CFLAGS=-Wextra -Wall -pedantic -std=c99 -g
LDFLAGS=-lm

default: test run

%.o: %.c
	$(CC) -c $< $(CFLAGS)

test: test.o csc.o spai.o
	$(CC) -o $@ $^ $(LDFLAGS)

run: test
	./test

clean:
	rm -rf spai test *.o