#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

#define RAND_FLOAT(min,max) \
    (((float)rand()/(float)(RAND_MAX/((max)-(min))))+(min))

#define SPEEDCOLOR_MAX 3.0
#define COLLISION_RADIUS 3
#define G 1

struct particle {
    float x; 
    float y;
    float vx;
    float vy;
    float mass;
    // Non-individual particles are simply skipped over.
    bool individual;
};

struct system {
    struct particle* particles;
    int size;
};

struct system gen_system( int size
                        , float min_x
                        , float max_x
                        , float min_y
                        , float max_y
                        , float min_vx
                        , float max_vx
                        , float min_vy
                        , float max_vy
                        , float min_mass
                        , float max_mass
                        )
{
    struct system sys;
    sys.size = size;
    sys.particles = malloc(sizeof(struct particle) * size);

    if (sys.particles == NULL) {
        fprintf(stderr, "ERROR: Could not allocate memory for particles.");
        exit(1);
    }

    srand(time(NULL));

    for (int i = 0; i < size; ++i) {
        struct particle p;

        p.x = RAND_FLOAT(min_x, max_x);
        p.y = RAND_FLOAT(min_y, max_y);
        p.vx = RAND_FLOAT(min_vx, max_vx);
        p.vy = RAND_FLOAT(min_vy, max_vy);
        p.mass = RAND_FLOAT(min_mass, max_mass);
        p.individual = true;
        sys.particles[i] = p;
    }

    return sys;
}

void calc_accel(struct particle* p1, struct particle* p2) {
    float dx = p2->x - p1->x;
    float dy = p2->y - p1->y;

    if (fabs(dx) <= COLLISION_RADIUS && fabs(dy) <= COLLISION_RADIUS) { 
        struct particle* max;
        struct particle* min;
        if (p1->mass > p2->mass) {
            max = p1;
            min = p2;
        } else {
            max = p2;
            min = p1;
        }

        float ms = p1->mass + p2->mass;
        min->individual = false;
        max->vx = (max->mass * max->vx + min->mass * min->vx) / ms;
        max->vy = (max->mass * max->vy + min->mass * min->vy) / ms;

        max->mass += min->mass;

        return;
    }

    // Hardcore optimization time!
    
    float d2 = dx*dx + dy*dy;
    float a1 = G * (p2->mass / d2);
    int sign = 1;

    if (dx > 0) {
        sign = 1;
    } else if (dx == 0) {
        if (dy > 0) {
            p1->vy += 1 * a1;
            p2->vy -= 1 * a1;
            return;
        } else {
            p1->vy -= 1 * a1;
            p2->vy += 1 * a1;
            return;
        }
    } else {
        sign = -1;
    }
    
    float dd = dy / dx;
    
    float ca = sign / sqrt(1 + dd*dd);
    float sa = dd * ca;

    p1->vx += a1 * ca;
    p1->vy += a1 * sa;

    float a2 = G * (p1->mass / d2);

    p2->vx -= a2 * ca;
    p2->vy -= a2 * sa;
}

void step_system(struct system* sys, SDL_Renderer* renderer) {
    int i;

    #pragma omp parallel private(i) shared(sys)
    {
        #pragma omp for
        for (i = 0; i < sys->size / 2; ++i) { 
            if (!sys->particles[i].individual)
                continue;

            for (int j = sys->size / 2; j < sys->size; ++j) {
                if (!sys->particles[j].individual)
                    continue;
                
                calc_accel(&sys->particles[i], &sys->particles[j]);
            }
        }
    }
        
    for (i = 0; i < sys->size; ++i) {
        if (!sys->particles[i].individual)
            continue;
        sys->particles[i].x += sys->particles[i].vx;
        sys->particles[i].y += sys->particles[i].vy;
        
        float off;
        float speed_sum =
            fabs(sys->particles[i].vx) + fabs(sys->particles[i].vy);

        if (speed_sum > SPEEDCOLOR_MAX) {
            off = 1.0;
        } else {
            off = speed_sum / SPEEDCOLOR_MAX;
        }
        
        uint8_t color = (uint8_t) (off * 255.0);

        SDL_SetRenderDrawColor(renderer, color, color, 255, 255);
        SDL_RenderDrawPoint(renderer, sys->particles[i].x, sys->particles[i].y);
    }
}

void usage(void) {
    fprintf(stderr, "USAGE: ./particles <trace?> <size>\n");
    exit(1);
}

int main(int argc, char** argv) {
    if (argc != 3) usage();
    
    bool trace = atoi(argv[1]);
    int size = atoi(argv[2]);

    if (size == 0) usage();
    
    printf("Generating particle system...\n"); 

    struct system sys =
        gen_system(size, 0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, 0, 0, 0, 0, 1, 10);

    printf("Creating window...\n");

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;

    window = SDL_CreateWindow( "Particles"
                             , SDL_WINDOWPOS_UNDEFINED
                             , SDL_WINDOWPOS_UNDEFINED
                             , SCREEN_WIDTH
                             , SCREEN_HEIGHT
                             , SDL_WINDOW_SHOWN | SDL_WINDOW_FULLSCREEN
                             );
    
    if (window == NULL) {
        fprintf(stderr, "ERROR: Couldn't create window. (%s)\n", SDL_GetError());
        exit(1);
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if (renderer == NULL) {
        fprintf(stderr, "ERROR: Couldn't create renderer (%s)\n", SDL_GetError()); 
        exit(1);
    }
    
    SDL_RenderClear(renderer); // Bit weird putting this here...

    SDL_Event event;
    bool running = true;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) 
                    running = false;
                else if (event.key.keysym.sym == SDLK_TAB) 
                    trace = !trace;
            }
        }
        
        if (!trace) {        
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
        }

        step_system(&sys, renderer);

        SDL_RenderPresent(renderer);
    }

    printf("Cleaning up...\n");
    
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

