all:
	gcc particles.c -o particles -Ofast -Wall -Wextra -pedantic -fopenmp -lSDL2 -lm

