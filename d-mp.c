#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <emmintrin.h>    // SSE2
#include <inttypes.h>
#include <time.h>

#define NANOS ((unsigned long)1000000000)
#define INIT_TICTAC() struct timespec tick, tock;
#define TICK() clock_gettime(CLOCK_MONOTONIC_RAW, &tick);
#define TOCK() clock_gettime(CLOCK_MONOTONIC_RAW, &tock);
#define PRINT_TT_DIFF(s) do {                                           \
        uint64_t diff = NANOS * (tock.tv_sec - tick.tv_sec) + tock.tv_nsec - tick.tv_nsec; \
        printf("%s%" PRIu64 "ns - %lfs\n", s, diff, (double)diff/NANOS); \
    } while(0);
#define TOCK_PRINT(s) do {TOCK(); PRINT_TT_DIFF(s);} while (0);


#ifdef DEBUG
#define LOG(...) fprintf(stderr, __VA_ARGS__)
#define VLOG(...) vprint(__VA_ARGS__)
#define VLLLOG(...) v_ll_print(__VA_ARGS__)
#else
#define LOG(...)
#define VLOG(...)
#define VLLLOG(...)
#endif

#define MAX_CAP 255 // Tipo uint8_t
#define VEC_TYPE __m128i
#define VEC_GT(va,vb) _mm_cmpgt_epi8(va, vb)
#define VEC_EQ(va,vb) _mm_cmpeq_epi8(va, vb)
#define VEC_MVMASK(va) _mm_movemask_epi8(va)
#define MAX_ELEMS (sizeof(VEC_TYPE))
#define ELEMS_MASK (MAX_ELEMS - 1)

typedef struct {
    int els; //n elements
    int span; //Number of integer values per element
    int elspan; //Each element's span (number of vectors)
    VEC_TYPE *elems;
} byte_vecc;

byte_vecc *alloc_byte_vecc(int els, int span) {
    byte_vecc *ret = malloc (sizeof(byte_vecc));
    ret->els = els;
    ret->span = span;
    ret->elspan = (span - 1) / MAX_ELEMS + 1;
    ret->elems = calloc(ret->els, ret->elspan * sizeof(VEC_TYPE));
    LOG("Alocando. Span: %d,  %lu bytes = els %d * elspan %d * sizeofvec %lu\n",
        ret->span,
        ret->els * ret->elspan * sizeof(VEC_TYPE),
        ret->els, ret->elspan, sizeof(VEC_TYPE));
    return ret;
}

void free_byte_vecc(byte_vecc *bv) {
    free(bv->elems);
    free(bv);
}

#define BVECC_AT0(bv, i, spanix) bv->elems[i * bv->elspan + (spanix)]
/* #define BVECC_AT(bv,i,j) BVECC_AT0(bv, i, (j >> ELEMS_MASK))[j & ELEMS_MASK] */


void fprint_byte_vecc(FILE *stream, byte_vecc *bv, char *prefix, char *sep, char*suffix) {
    for (int i = 0; i < bv->els; ++i) {
        fprintf(stream, "%s", prefix);
        int remaining = bv->span;
        for (int j = 0;  j < bv->elspan; ++ j) {
            uint8_t *val = (uint8_t*) &BVECC_AT0(bv, i, j);
            int k;
            for (k = 0; k < remaining && k < (int)MAX_ELEMS; ++k) {
                fprintf(stream, "%d%s", val[k], (k == remaining - 1) ? suffix : sep);
            }
            remaining -= k;
        }
        assert (remaining == 0);
    }
}

void print_byte_vecc(byte_vecc *bv, char *prefix, char *sep, char* suffix) {
    fprint_byte_vecc(stdout, bv, prefix, sep, suffix);
}


#define MIN(X,Y) (X < Y ? X : Y)
#define MAX(X,Y) (X > Y ? X : Y)

int *vcopy(int n, const int *v) {
    int bytes = sizeof(int) * n;
    int *ret = malloc (bytes);
    memcpy(ret, v, bytes);
    return ret;
}

int vmax (int n, int *v) {
    int max = INT_MIN;
    for (int i = 0; i < n; ++i)
        if (v[i] > max)
            max = v[i];
    return max;
}

void vsub (int n, int *v1, const int *v2) {
    for (int i = 0; i < n; ++i)
        v1[i] -= v2[i];
}

int vequals (int n, const int *v1, const int * v2) {
    for (int i = 0; i < n; ++i)
        if (v1[i] != v2[i])
            return 0;
    return 1;
}

int vleq_sort(int n, const int *v1, const int * v2) {
    int i = n - 1;
    while (i >= 0) {
        int d1 = v1[i];
        int d2 = v2[i];
        if (d1 < d2) return 1;
        if (d1 > d2) return 0;
        i--;
    };
    return 1;
}

int vless (int n, const int *v1, const int *v2) {
    int ret = 0;
    for (int i = 0; i < n; ++i) {
        if (v1[i] > v2[i])
            return 0;
        ret |= v1[i] != v2[i];
    }
    return ret;
}

void fvprint(FILE *stream, char* pref, int n, const int *v, char *sep, char *suf) {
    fprintf(stream, "%s", pref);
    for (int i = 0; i < n; ++i) {
        fprintf (stream, "%d", v[i]);
        if (i + 1 < n)
            fprintf (stream, "%s", sep);
    }
    fprintf(stream, "%s", suf);
}

void vprint (char* pref, int n, const int *v, char *suf) {
    fvprint(stdout, pref, n, v, " ", suf);
}

typedef struct _struct_v_ll {
    //para a cabeça indica o numero de elementos na lista
    int n;
    int *v;
    struct _struct_v_ll *next;
} v_ll;

v_ll *v_ll_create () {
    v_ll *head = malloc (sizeof(v_ll));
    head->n = 0;
    head->v = NULL;
    head->next = NULL;
    return head;
}

//Assume que é responsável por liberar os vetores
void free_v_ll(v_ll *head) {
    v_ll *curr = head->next;
    while (curr) {
        v_ll *this = curr;
        curr = this->next;
        free(this->v);
        free(this);
    }
    free(head);
}

void v_ll_insert(int n, int *v, v_ll *head) {
    v_ll *cell = malloc (sizeof(v_ll));
    cell->n = n;
    cell->v = v;
    cell->next = head->next;
    head->next = cell;
    head->n++;
}

void v_ll_print(char *prefix, v_ll *head) {
    v_ll *cur = head->next;
    int i = 0;
    while (cur) {
        printf("%s%d: ", prefix, i++);
        vprint("", cur->n, cur->v, "\n");
        cur = cur->next;
    }
}

int **v_ll_to_vec (v_ll *head) {
    int ** ret = malloc (head->n * sizeof(int*));
    v_ll *curr = head->next;
    int pos = 0;
    while (curr) {
        ret[pos] = curr->v;
        pos++;
        curr = curr->next;
    }
    return ret;
}

byte_vecc *v_ll_to_byte_vec (v_ll *head, int vector_length) {
    byte_vecc *ret = alloc_byte_vecc(head->n, vector_length);
    v_ll *curr = head->next;
    int pos = 0;
    while (curr) {
        for (int i = 0; i < vector_length; ++ i) {
            uint8_t *vs = (uint8_t*)(&BVECC_AT0(ret, pos, i >> 4));
            vs[i & ELEMS_MASK] = curr->v[i];
            LOG("Set> Pos %d Span %d vs[%d] = %d\n",
                pos, i >> 4, i & (int)ELEMS_MASK, curr->v[i]);
        }
        pos++;
        curr = curr->next;
    }
    return ret;
}

typedef struct {
    char name[11];
    int id;
    int visited;
    struct _struct_edge_ll *edges;
} node;

typedef struct _struct_edge {
    int id;
    int from;
    int to;
} edge;

typedef struct {
    int n;
    node* nodes;
    int m;
    edge *edges;
    int *edge_capacities;
} graph;

typedef struct _struct_edge_ll {
    int edge;
    struct _struct_edge_ll* next;
} edge_ll;

typedef struct _path_list {
    edge_ll* path;
    int cap;
    struct _path_list* next;
} path_ll;

typedef struct {
    int path_count;
    int *path_caps;
    path_ll *paths;

    int *nvpaths;
    int **vpaths;
} amps;

void print_edge(graph *g, int e) {
    printf("%s->%s ",
           g->nodes[g->edges[e].from].name,
           g->nodes[g->edges[e].to].name);
}

//Paths are stored in reverse order
void print_path (graph *g, edge_ll *p) {
    if (!p) return;
    print_path(g, p->next);
    printf (" ");
    print_edge(g, p->edge);
}

void print_path_list (graph *g,  amps *mps) {
    path_ll *cplist = mps->paths;
    int n = 0;
    while (cplist) {
        printf("Path %d (Ln %d Cp %d): ", n, mps->nvpaths[n], cplist->cap);
        print_path(g, cplist->path);
        cplist = cplist->next;
        printf("\n");
        n++;
    }
}

void print_amps(graph *g, amps *mps) {
    (void)g;
    printf("Total paths: %d\n", mps->path_count);
    for (int i = 0; i < mps->path_count; ++i) {
        LOG("Path %d (Ln %d Cp %d): ", i, mps->nvpaths[i], mps->path_caps[i]);
        for (int j = 0; j < mps->nvpaths[i]; ++j) {
            LOG("a_%d ", mps->vpaths[i][j]);
        }
        LOG("\n");
    }
#ifdef DEBUG
    print_path_list(g, mps);
#endif
}

void print_node (graph *g, int node) {
    printf("N%s: ", g->nodes[node].name);
    print_path(g, g->nodes[node].edges);
}

void print_graph(graph *g) {

    printf("nodes: %d\n", g->n);
    printf("edges: %d\n", g->m);
    for (int i = 0; i < g->n; ++i) {
        print_node(g, i);
        printf("\n");
    }
    printf("caps: ");
    for (int i = 0; i < g->m; ++i)
        printf("%d ", g->edge_capacities[i]);
    printf("\n");
}

edge_ll *path_include_edge(graph *g, edge_ll *path, int edge) {
    g->nodes[g->edges[edge].to].visited = 1;
    edge_ll *cell = (edge_ll*)malloc(sizeof(edge_ll));
    cell->edge = edge;
    cell->next = path;
    return cell;
}

edge_ll * path_remove_edge(graph *g, edge_ll *cpath) {
    g->nodes[g->edges[cpath->edge].to].visited = 0;
    edge_ll *ret = cpath->next;
    free(cpath);
    return ret;
}

edge_ll *duplicate_path(edge_ll *path) {
    if (!path)
        return NULL;
    edge_ll *ret = (edge_ll*)malloc(sizeof(edge_ll));
    ret->edge = path->edge;
    ret->next = NULL;
    edge_ll *nlist = ret;
    edge_ll *curr = path->next;
    while (curr) {
        nlist->next = (edge_ll*)malloc(sizeof(edge_ll));
        nlist = nlist->next;
        nlist->edge = curr->edge;
        curr = curr->next;
    }
    nlist->next = NULL;
    return ret;
};

void path_list_include_path (amps *mps, edge_ll* path) {
    edge_ll *path_copy = duplicate_path(path);
    path_ll *newPath = malloc(sizeof(path_ll));
    newPath->path = path_copy;
    newPath->next = mps->paths;
    mps->path_count++;
    mps->paths = newPath;
}

void all_minimal_paths_e(graph *g, int start, int end, amps *mps, edge_ll *cpath) {
    if (start == end)
        path_list_include_path(mps, cpath);
    edge_ll *npath = cpath;
    edge_ll *curr_e = g->nodes[start].edges;
    while (curr_e) {
        if (!g->nodes[g->edges[curr_e->edge].to].visited) {
            npath = path_include_edge(g, npath, curr_e->edge);
            all_minimal_paths_e(g, g->edges[curr_e->edge].to, end, mps, npath);
            npath = path_remove_edge(g, npath);
        }
        curr_e = curr_e->next;
    }
}


// TODO - É possível juntar calculate_capacity e fill_vpath
void calculate_capacity(graph *g, edge_ll *path, int *capacity, int *length) {
    int max_cap = INT_MAX;
    int l = 0;
    edge_ll *c = path;
    while (c) {
        l++;
        int cap = g->edge_capacities[g->edges[c->edge].id];
        if (cap < max_cap)
            max_cap = cap;
        c = c->next;
    }
    *capacity = max_cap;
    *length = l;
}

void fill_vpath(int *v, graph *g, edge_ll *path, int len) {
    edge_ll *c = path;
    int i = len - 1;
    while (c) {
        v[i] = g->edges[c->edge].id;
        i--;
        c = c->next;
    }
}

amps *all_minimal_paths(graph *g, int start, int end) {

    amps *mps = malloc (sizeof(amps));
    mps->path_count = 0;
    mps->paths = NULL;
    mps->path_caps = NULL;

    g->nodes[start].visited = 1;
    all_minimal_paths_e(g, start, end, mps, NULL);
    g->nodes[start].visited = 0;

    mps->path_caps = malloc(sizeof(int) * mps->path_count);
    mps->nvpaths = malloc(sizeof(int) * mps->path_count);
    mps->vpaths = malloc(sizeof(int*) * mps->path_count);
    path_ll *curr = mps->paths;
    int i = 0;
    while (curr) {
        int length, cap;
        calculate_capacity(g, curr->path, &cap, &length);
        curr->cap = cap;
        mps->path_caps[i] = cap;
        mps->nvpaths[i] = length;
        mps->vpaths[i] = malloc(sizeof(int) * length);
        fill_vpath(mps->vpaths[i], g, curr->path, length);
        curr = curr->next;
        i++;
    }

    print_amps(g, mps);

    return mps;
}



#define NAME_IDX(X) (X - 1)
#define IDX_ID(X) (X + 1)


void graphIncludeDirEdge(graph *g, int edge_id, int arc_id, int from_name, int to_name) {
    node *nodes = g->nodes;
    edge *edges = g->edges;

    edges[arc_id].id = edge_id;
    edges[arc_id].from = NAME_IDX(from_name);
    edges[arc_id].to = NAME_IDX(to_name);

    edge_ll *cell = malloc(sizeof(edge_ll));
    cell->edge = arc_id;
    cell->next = nodes[NAME_IDX(from_name)].edges;
    nodes[NAME_IDX(from_name)].edges = cell;
}

graph *init_graph(int dim, int edges, int *caps, int *srcs, int *dests) {
    graph *g = malloc(sizeof(graph));

    g->n = dim;
    g->nodes = malloc(sizeof(node) * dim);
    for (int i = 0; i < dim; ++i) {
        sprintf(g->nodes[i].name, "%d", IDX_ID(i));
        g->nodes[i].id = i;
        g->nodes[i].edges = NULL;
        g->nodes[i].visited = 0;
    }

    g->m = edges; //undirected represented as 2 arcs
    g->edges = malloc(sizeof(edge) * g->m * 2);
    for (int i = 0; i < edges; ++i) {
        graphIncludeDirEdge(g, i, 2 * i, srcs[i],  dests[i]);
        graphIncludeDirEdge(g, i, 2 * i + 1, dests[i], srcs[i] );
    }
    g->edge_capacities = malloc(sizeof(int) * edges);
    for (int i = 0; i < edges; i++) {
        g->edge_capacities[i] = caps[i];
    }
    return g;
}

void free_edge(edge *e) {
    free(e);
}

void free_edge_ll(edge_ll *cell) {
    if (cell->next)
        free_edge_ll(cell->next);
    free(cell);
}

void free_node_contents(node *n) {
    free_edge_ll(n->edges);
}

void free_graph(graph *g) {
    for (int i = 0; i < g->n; ++i)
        free_node_contents(&g->nodes[i]);
    free(g->nodes);
    free(g->edges);
    free(g->edge_capacities);
    free(g);
}

void free_path_list(path_ll *pl) {
    free_edge_ll(pl->path);
    if (pl->next)
        free_path_list(pl->next);
    free(pl);
}

void free_amps(amps *mps) {
    free_path_list(mps->paths);
    free(mps->path_caps);
    free(mps->nvpaths);
    for (int i = 0; i < mps->path_count; ++i) {
        free(mps->vpaths[i]);
    }
    free(mps->vpaths);
    free(mps);
}

// true se viável
int include_path_cap_to_solution (amps *mps, int cap, int path, int *sol, int *M) {
    int ret = 1;
    sol[path] = cap;

    for (int i = 0; i < mps->nvpaths[path]; ++i) {
        M[mps->vpaths[path][i]] -= cap;
        ret &= M[mps->vpaths[path][i]] >= 0;
    }

    return ret;
}

void remove_path_cap_to_solution (amps *mps, int cap, int path, int *sol, int *M) {
    sol[path] = 0;
    for (int i = 0; i < mps->nvpaths[path]; ++i)
        M[mps->vpaths[path][i]] += cap;
}

void feasible_flow_solutions_for (amps *mps,
                                  int d,
                                  int m, int *M, const int *pristineM,
                                  int npaths, int *pathCaps, int *currentSol,
                                  int index,
                                  v_ll *sols_head) {//sols: ll com cabeça
    LOG("FFS d=%d npaths=%d index=%d\n", d, npaths, index);
    assert(d >= 0);

    if (d == 0) {
        int *found = vcopy(m, pristineM);
        vsub(m, found, M);
        v_ll_insert(m, found, sols_head);
        VLOG("PS ", npaths, currentSol, " ");
        VLOG("FrM ", m, M, " ");
        VLOG("UsM ", m, found, "\n");
    } else if (index >= npaths) {
        LOG("FFS Saindo A\n");
        return;
    } else {
        //Ainda há espaço
        //Garante cond. 1 da eq. diofantina
        int max_in_pos = MIN (pathCaps[index], d);
        for (int v = max_in_pos; v >= 0; v--) {
            //Garante cond. 2 da eq. diofantina
            if (include_path_cap_to_solution(mps, v, index, currentSol, M))
                feasible_flow_solutions_for(
                                            mps,
                                            d - v,
                                            m, M, pristineM,
                                            npaths, pathCaps, currentSol,
                                            index + 1,
                                            sols_head);
            remove_path_cap_to_solution(mps, v, index, currentSol, M);
        }
    }
}

v_ll *feasible_flow_solutions (graph *g, amps *mps, int d) {
    int m = g->m;
    int *M = vcopy(m, g->edge_capacities);

    int npaths = mps->path_count;
    int *pathCaps = vcopy(npaths, mps->path_caps);

    v_ll *sol_head = v_ll_create();
    int *acc = calloc(npaths, sizeof(int));

    LOG("m = %d\n", m);
    VLOG("M = ", m, M, "\n");
    LOG("maxD = %d \n", vmax(m, M));
    LOG("npaths = %d \n", npaths);
    VLOG("pathCaps = ", npaths, pathCaps, "\n");
    LOG("\n> Iniciando calculo para d = %d\n", d);

    feasible_flow_solutions_for (mps,
                                 d,
                                 m, M, g->edge_capacities,
                                 npaths, pathCaps, acc,
                                 0,
                                 sol_head);

    free(M);
    free(pathCaps);
    free(acc);

    return sol_head;
}

// Recebe vetor v[p..r] com p <= r. Rearranja
// os elementos do vetor e devolve j em p..r
// tal que v[p..j-1] <= v[j] < v[j+1..r].

static int partition (int** v, int p, int r, int n, int* indices) {
    int c = indices[r]; // pivô
    int t, j = p;
    for (int k = p; k < r; ++k) {
        if (vleq_sort(n, v[indices[k]], v[c])) {
            t = indices[j], indices[j] = indices[k], indices[k] = t;
            ++j;
        }
    }
    t = indices[j], indices[j] = indices[r], indices[r] = t;
    return j;
}


void quickSort (int** v, int p, int r, int n, int* indices) {
    while (p < r) {
        int j = partition (v, p, r, n, indices);
        if (j - p < r - j) {
            quickSort (v, p, j - 1, n, indices);
            p = j + 1;
        } else {
            quickSort (v, j + 1, r, n, indices);
            r = j - 1;
        }
    }
}


int **remove_duplicates (graph * g, v_ll *xs_l, int* nxs) {

    int ls = xs_l->n;
    int cs = g->m;
    printf("ls = %d cs = %d\n", ls, cs);

    int **xs = v_ll_to_vec(xs_l);

    int *sortPos = malloc(ls * sizeof(int));
    for (int i = 0; i < ls; ++i)
        sortPos[i] = i;

    printf("Sorting...");
    fflush(stdout);
    quickSort(xs, 0, ls - 1, cs, sortPos);
    printf("done!\n");
    LOG("Ordem: ");
    for (int i = 0; i < ls; ++i)
        LOG("%d ", sortPos[i]);
    LOG("\n");

    int remaining = ls;
    int *remove = calloc(ls, sizeof(int));
    int last = 0;
    for (int i = 1; i < ls; i++) {
        if (vequals(cs, xs[sortPos[last]], xs[sortPos[i]])) {
            LOG("Removendo %d = %d\n", sortPos[last], sortPos[i]);
            remove[sortPos[i]] = 1;
            remaining--;
        } else {
            last = i;
        }
    }
    free(sortPos);
    printf("Tot. %d Dups. %d Remaining: %d\n", ls, ls - remaining, remaining);


    int pos = 0;
    int **ret = malloc (sizeof(int*) * remaining);
    for (int i = 0; i < ls; i++) {
        if (!remove[i]) {
            ret[pos] = vcopy(cs, xs[i]);
            pos++;
        }
    }

    free(remove);
    free(xs);
    *nxs = remaining;
    return ret;
}

void print128_num(__m128i var)
{
    uint8_t *val = (uint8_t*) &var;
    for (int i = 0; i < 16; ++i) {
        printf("%i ", val[i]);
    }
}

int vequals2 (byte_vecc* bvcc, const int v1, const int v2) {
    for (int i = 0; i < bvcc->elspan; ++i) {
        VEC_TYPE d1 = BVECC_AT0(bvcc, v1, i);
        VEC_TYPE d2 = BVECC_AT0(bvcc, v2, i);
        VEC_TYPE eq = VEC_EQ(d1, d2);
        int ieq = VEC_MVMASK(eq) == 0xffff;
        if (!ieq)
            return 0;
    }
    return 1;
}

int vleq_sort2(byte_vecc* bv, const int p1, const int p2) {
    int comps = 0;
    int i = bv->elspan - 1;
    while (i >= 0) {

        VEC_TYPE d1 = BVECC_AT0(bv, p1, i);
        VEC_TYPE d2 = BVECC_AT0(bv, p2, i);
        VEC_TYPE diff = d1 - d2;

        int lim = MIN(bv->span - comps, (int)MAX_ELEMS);
        for (int j = 0; j < lim; j++) {
            if (diff[j] < 0) return 1;
            if (diff[j] > 0) return 0;
        }
        comps += lim;
        i--;
    };
    return 1;
}

int vless2 (byte_vecc* bv, const int p1, const int p2) {
    int ret = 0;
    int i = bv->elspan - 1;
    while (i >= 0) {
        VEC_TYPE d1 = BVECC_AT0(bv, p1, i);
        VEC_TYPE d2 = BVECC_AT0(bv, p2, i);
        VEC_TYPE gt = VEC_GT(d1, d2);
        int gtmask = VEC_MVMASK(gt);
        /* printf("d1(%d): ", p1); print128_num(d1); */
        /* printf("\nd2(%d): ", p2); print128_num(d2); */
        /* printf("\ngt: "); print128_num(gt); */
        /* printf("\ngtmask = %d %x\n", gtmask, gtmask); */
        if (gtmask) return 0;
        VEC_TYPE diff = d1 - d2;
        int diffMask = VEC_MVMASK(diff);
        /* printf("diff: "); print128_num(diff); */
        /* printf("\ndiffmask = %d %x\n", diffMask, diffMask); */
        ret |= diffMask;
        i--;
    };
    return ret;
}


static int partition2 (byte_vecc* bvcc, int p, int r, int* indices) {
    int c = indices[r]; // pivô
    int t, j = p;
    for (int k = p; k < r; ++k) {
        if (vleq_sort2(bvcc, indices[k], c)) {
            t = indices[j], indices[j] = indices[k], indices[k] = t;
            ++j;
        }
    }
    t = indices[j], indices[j] = indices[r], indices[r] = t;
    return j;
}


void quickSort2 (byte_vecc* bvcc, int p, int r, int* indices) {
    while (p < r) {
        int j = partition2 (bvcc, p, r, indices);
        if (j - p < r - j) {
            quickSort2 (bvcc, p, j - 1, indices);
            p = j + 1;
        } else {
            quickSort2 (bvcc, j + 1, r, indices);
            r = j - 1;
        }
    }
}


byte_vecc *remove_duplicates2 (graph * g, v_ll *xs_l) {

    int ls = xs_l->n;
    LOG("ls = %d cs = %d\n", ls, g->m);

    byte_vecc *xs = v_ll_to_byte_vec(xs_l, g->m);

    int *sortPos = malloc(ls * sizeof(int));
    for (int i = 0; i < ls; ++i)
        sortPos[i] = i;

    LOG("Sorting...");
    fflush(stdout);
    quickSort2(xs, 0, ls - 1, sortPos);
    LOG("done!\n");

    LOG("Ordem: ");
    for (int i = 0; i < ls; ++i)
        LOG("%d ", sortPos[i]);
    LOG("\n");

    int remaining = ls;
    int *remove = calloc(ls, sizeof(int));
    int last = 0;
    for (int i = 1; i < ls; i++) {
        if (vequals2(xs, sortPos[last], sortPos[i])) {
            LOG("Removendo %d = %d\n", sortPos[last], sortPos[i]);
            remove[sortPos[i]] = 1;
            remaining--;
        } else {
            last = i;
        }
    }
    free(sortPos);
    printf("Tot. %d Dups. %d Remaining: %d\n", ls, ls - remaining, remaining);

    byte_vecc *ret = alloc_byte_vecc(remaining, g->m);
    assert(ret->els == remaining);
    assert(ret->span == xs->span);
    assert(ret->elspan == xs->elspan);

    int pos = 0;
    for (int i = 0; i < ls; i++) {
        if (!remove[i]) {
            for(int j = 0; j < ret->elspan; j++) {
                BVECC_AT0(ret, pos, j) = BVECC_AT0(xs, i, j);
            }
            pos++;
        }
    }
    free(remove);
    free_byte_vecc(xs);

    return ret;
}

void remove_non_minimal (graph * g, int *nxs, int ***xss) {
    int **xs = *xss;
    int   ls = *nxs;
    int   cs = g->m;
    int *remove = calloc(ls, sizeof(int));
    int remaining = ls;

    int countRemoved = 0;
#pragma omp parallel for default (none)                     \
    reduction (+: countRemoved) shared (remove, cs, xs, ls)
    for (int i = 0; i < ls; i++) {
        if (remove[i]) continue;
        for (int j = i + 1; j < ls; j++) {
            if (remove[j]) continue;
            //since we already have i and j in cache, check both ways
            if (vless(cs, xs[j], xs[i])) {
                int removed =  __sync_bool_compare_and_swap(&remove[i], 0, 1);
                if (removed) {
                    //Only counts if removed
                    LOG("Removendo %d pois xs[%d] < xs[%d]\n", i, j, i);
                    countRemoved++;
                }
                break; // breaks anyway since i is no longer relevant
            } else if (vless(cs, xs[i], xs[j])) {
                int removed =  __sync_bool_compare_and_swap(&remove[j], 0, 1);
                if (removed) {
                    LOG("Removendo %d pois xs[%d] < xs[%d]\n", j, i, j);
                    countRemoved++;
                }
                //In this case cannot break, since j was removed. Not i.
            }
        }
    }
    remaining -= countRemoved;

    printf("Tot. %d Removed: %d Remaining: %d\n", ls, ls - remaining, remaining);

    int pos = 0;
    int **ret = malloc (sizeof(int*) * remaining);
    for (int i = 0; i < ls; i++) {
        if (remove[i]) {
            free(xs[i]);
        } else {
            ret[pos] = xs[i];
            pos++;
        }
    }
    free(remove);
    free(xs);
    assert(pos == remaining);
    *xss = ret;
    *nxs = pos;
}


byte_vecc *remove_non_minimal2 (byte_vecc *bv) {
    int n = bv->els;
    int *remove = calloc(n, sizeof(int));
    int countRemoved = 0;

#pragma omp parallel for default (none)                 \
    reduction (+: countRemoved) shared (remove, n, bv)
    for (int i = 0; i < n; i++) {
        if (remove[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (remove[j]) continue;
            //since we already have i and j in cache, check both ways
            if (vless2(bv, j, i)) {
                int removed =  __sync_bool_compare_and_swap(&remove[i], 0, 1);
                if (removed) {
                    //Only counts if removed
                    LOG("Removendo %d pois xs[%d] < xs[%d]\n", i, j, i);
                    countRemoved++;
                }
                break; // breaks anyway since i is no longer relevant
            } else if (vless2(bv, i, j)) {
                int removed =  __sync_bool_compare_and_swap(&remove[j], 0, 1);
                if (removed) {
                    LOG("Removendo %d pois xs[%d] < xs[%d]\n", j, i, j);
                    countRemoved++;
                }
                //In this case cannot break, since j was removed. Not i.
            }
        }
    }
    int remaining = n - countRemoved;

    printf("Tot. %d Removed: %d Remaining: %d\n", n, countRemoved, remaining);


    byte_vecc *ret = alloc_byte_vecc(remaining, bv->span);
    assert(ret->els == remaining);
    assert(ret->span == bv->span);
    assert(ret->elspan == bv->elspan);

    int pos = 0;
    for (int i = 0; i < n; i++) {
        if (!remove[i]) {
            for(int j = 0; j < ret->elspan; j++) {
                BVECC_AT0(ret, pos, j) = BVECC_AT0(bv, i, j);
            }
            pos++;
        }
    }
    assert(pos == remaining);
    free(remove);
    free_byte_vecc(bv);

    return ret;
}


graph *from_matrix_gen(int dim, int m[dim][dim]) {
    int edges = 0;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < i; ++j) //Assume matriz simétrica
            edges += !!m[i][j];

    int srcs[edges];
    int dests[edges];
    int caps[edges];
    int pos = 0;
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < i; ++j) //Assume matriz simétrica
            if (m[i][j]) {
                srcs[pos] = i + 1; // matriz 1-based
                dests[pos] = j + 1;
                caps[pos] = m[i][j];
                pos++;
            }
    assert(pos == edges);

    return init_graph(dim, edges, caps, srcs, dests);
}

graph *from_file_gen(char *fname, int cap) {
    FILE *f = fopen(fname, "r");
    assert(f);

    char c = 0;
    int dim = 1;
    while (c != '\n') {
        fscanf(f, "%c", &c);
        if (c == ',')
            dim++;
    }
    rewind(f);

    int m[dim][dim];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; ++j) {
            int r = fscanf(f, "%d,", &m[i][j]);
            assert(r == 1);
            if (m[i][j] != 0 && cap != -1)
                m[i][j] = cap;
        }
    }
    fclose(f);

    return from_matrix_gen(dim, m);
}

int main(int argc, char **argv) {
    INIT_TICTAC();

    struct timespec inicio, fim;

    if (argc < 3 || argc > 5) {
        printf("Erro. Use: prog bench_file d [cap] [0/1 DO NOT/DO use vectorization]\n");
        exit(1);
    }


    int use_vec = 1;
    if (argc == 5) use_vec = atoi(argv[4]);

    if (use_vec) {
        printf("Vectorization options: MaxCap: %d VectorSize: %ld\n",
               MAX_CAP, MAX_ELEMS);
    } else {
        printf("Not using vectorization\n");
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &inicio);

    printf("Step 0 - Graph creation\n------------\n");
    TICK();
    int d = atoi(argv[2]);
    int cap = argc >= 4 ? atoi(argv[3]) : -1;
    printf("d=%d\n", d);
    graph *g = from_file_gen(argv[1], cap);
    print_graph(g);
    TOCK_PRINT("Tempo total: ");


    printf("\n\n\nStep 1 - Find all MPs\n------------\n");
    TICK();
    amps *mps = all_minimal_paths(g, 0, g->n - 1);
    TOCK_PRINT("Tempo total: ");


    printf("\n\n\nStep 2 - Solve diophantine equations\n------------\n");
    TICK();
    v_ll *xs_l = feasible_flow_solutions(g, mps, d);
    printf("Solutions found: %d\n", xs_l->n);
    TOCK_PRINT("Tempo total: ");


    printf("\n\n\nStep 3 - Remove duplicates\n------------\n");
    int nxs;
    int **xs = NULL;
    byte_vecc *xs2 = NULL;
    TICK();
    if (use_vec) {
        xs2 = remove_duplicates2(g, xs_l);
    } else {
        xs = remove_duplicates(g, xs_l, &nxs);
    }
    free_v_ll(xs_l);
    TOCK_PRINT("Tempo total: ");


    printf("\n\n\nStep 4 - Remove non-minimal\n------------\n");
    TICK();
    if (use_vec) {
        xs2 = remove_non_minimal2(xs2);
    } else {
        remove_non_minimal(g, &nxs, &xs);
    }
    TOCK_PRINT("Tempo total: ");

    printf("\n\n\nStep 5 - Write output\n------------\n");
    TICK();
    if (use_vec) {
        FILE *fp = fopen("output_v.out", "w");
        fprint_byte_vecc(fp, xs2, "", " ", "\n");
        fclose(fp);
        free_byte_vecc(xs2);
    } else {
        FILE *fp = fopen("output.out", "w");
        for (int i = 0; i < nxs; ++i) {
            fvprint(fp, "", g->m, xs[i], " ", "\n");
        }
        fclose(fp);
        for (int i = 0; i < nxs; ++i)
            free(xs[i]);
        free(xs);
    }
    free_amps(mps);
    free_graph(g);
    TOCK_PRINT("Tempo total: ");

    clock_gettime(CLOCK_MONOTONIC_RAW, &fim);
    uint64_t tot = NANOS * (fim.tv_sec - inicio.tv_sec) + fim.tv_nsec - inicio.tv_nsec;
    printf("> %i %i %i %lfs\n", d, cap, use_vec, (double)tot/NANOS);

    return 0;
}
