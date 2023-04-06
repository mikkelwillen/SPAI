CC=gcc
CFLAGS=-Wextra -Wall -pedantic -std=c99 -g
LDFLAGS=-lm

all: 

%.o: %.c
	$(CC) -c $< $(CFLAGS)

test: test.o csc.o spai.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf spai