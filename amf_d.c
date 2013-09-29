/* Author: Zhengyu He (hezhengyu@gmail.com) */

#include <time.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "amf.h"
#include "atomic.h"

int num_threads;
int g_num_nodes;
int g_num_edges;
long totalPushes;
long totalLifts;
int gr_threshold;
int GRcnt = 0;
int HPcnt = 0;

pthread_mutex_t gr_mutex;
pthread_mutex_t* thread_mutex;
pthread_mutex_t* node_mutex;
pthread_barrier_t start_barrier;

struct node_entry {
  int height;
  volatile int inQ;
  volatile long excess;
  int wave;
  struct edge_entry* adj_list;
};

typedef struct node_entry Node;

struct edge_entry {
  Node* endpoint;
  long capacity;
  struct edge_entry* nextedge;
  struct edge_entry* mateedge;
};

typedef struct edge_entry* Edge;

typedef struct {
  int Qsize;
  int head;
  int tail;
  volatile int request;
  volatile int shareAmount;
  Node** Q;
} ThreadEnv;

Node* g_node;
Node* source, *sink;
Node* NoNode;
ThreadEnv* threadEnv;
Edge edge_sym, edge_seq;
int edgecnt;
Edge search_adj(Node* nodeid, Edge adjL);
Edge Addedge(int from, int to, int capacity_f, int capacity_b, Edge edge_arr);

inline bool flow_done(){
  return source->excess + sink->excess >= 0;
}

int check_violation() {
  Node* i;
  Edge edge;
  for (i = g_node; i < g_node + g_num_nodes; i++) {

    for (edge = i->adj_list; edge < (i + 1)->adj_list; edge++) {
      if (edge->capacity > 0 && i->height > edge->endpoint->height + 1) {
        fprintf(stderr, "violation found!\n");
        fprintf(stderr,
                "%ld(%d)(w %d) -%ld-> %ld(%d)(w %d) \n",
                i - g_node,
                i->height,
                i->wave,
                edge->capacity,
                edge->endpoint - g_node,
                edge->endpoint->height,
                edge->endpoint->wave);
        fflush(stderr);
      }
    }
  }
  return 0;
}

Edge Addedge(int from, int to, int capacity_f, int capacity_b, Edge edge_arr) {
  Edge edge_f, edge_b;

  edge_f = edge_arr;
  edge_b = edge_f + 1;

  if (from == to)
    return edge_arr;

  if (from < to) {  // from is small
    edgecnt += 2;
    // add edge to from.adj_list
    edge_f->capacity = capacity_f;
    edge_f->endpoint = g_node + to;
    edge_f->mateedge = edge_b;
    edge_f->nextedge = g_node[from].adj_list;

    g_node[from].adj_list = edge_f;

    // add edge to to.adj_list
    edge_b->capacity = capacity_b;
    edge_b->endpoint = g_node + from;
    edge_b->mateedge = edge_f;
    edge_b->nextedge = g_node[to].adj_list;

    g_node[to].adj_list = edge_b;

  } else {  // to is small

    // search to->adj_list
    edge_f = search_adj(g_node + from, g_node[to].adj_list);
    // edge_f =  search_adj(g_node+to, g_node[from].adj_list );

    if (edge_f != NULL) {

      // from->to does exist. update its capacity
      edge_f->mateedge->capacity += capacity_f;
      return edge_arr;

    } else {
      return Addedge(to, from, capacity_b, capacity_f, edge_arr);
    }
  }

  return edge_arr + 2;
}

Edge search_adj(Node* nodeid, Edge adjL) {
  Edge edge;
  edge = adjL;
  if (edge == NULL)
    return NULL;

  // printf("%d %d\n",nodeid-g_node, adjL);
  while (edge->endpoint != nodeid) {
    if (edge->nextedge != NULL)
      edge = edge->nextedge;
    else
      return NULL;
  }

  return edge;
}
int SortEdges() {

  edge_seq = (Edge)malloc(edgecnt * sizeof(struct edge_entry));
  if (edge_seq == NULL)
    exit(-1);
  int n = 0;
  Node* i;
  Edge temp, e;
  for (i = g_node; i < g_node + g_num_nodes; i++) {
    e = i->adj_list;
    i->adj_list = edge_seq + n;
    while (e != NULL) {
      edge_seq[n].endpoint = e->endpoint;
      edge_seq[n].capacity = e->capacity;

      if (i > e->endpoint) {
        edge_seq[n].mateedge = e->mateedge->mateedge;
        edge_seq[n].mateedge->mateedge = edge_seq + n;
      } else {
        e->mateedge = edge_seq + n;
      }
      n++;
      e = e->nextedge;
    }
  }
  i->adj_list = edge_seq + n;
  free(edge_sym);
  return 0;
}
//==============queue=========
inline int reenQ(ThreadEnv* t, Node* nodeadd) {
  t->Q[t->tail] = nodeadd;
  t->Q[t->head] = NoNode;
  if (t->tail < LOCAL_Q_SIZE - 1)
    t->tail++;
  else
    t->tail = 0;

  if (t->head < LOCAL_Q_SIZE - 1)
    t->head++;
  else
    t->head = 0;

  return 0;
}

inline int enQ(ThreadEnv* t, Node* nodeadd) {
  t->Q[t->tail] = nodeadd;
  if (t->tail < LOCAL_Q_SIZE - 1)
    t->tail++;
  else
    t->tail = 0;
  t->Qsize++;
  return 0;
}

inline int deQ(ThreadEnv* t, Node* nodeadd) {

  t->Qsize--;
  t->Q[t->head] = NoNode;
  if (t->head < LOCAL_Q_SIZE - 1)
    t->head++;
  else
    t->head = 0;

  return 0;
}
//==============queue=========

int init_threads(int n) {
  int i;
  threadEnv = (ThreadEnv*)malloc(sizeof(ThreadEnv) * num_threads);
  if (threadEnv == NULL)
    exit(-1);
  for (i = 0; i < n; i++) {
    threadEnv[i].Qsize = threadEnv[i].head = threadEnv[i].tail = 0;  //??
    threadEnv[i].Q = (Node**)calloc(Q_SIZE, sizeof(Node*));
    if (threadEnv[i].Q == NULL)
      exit(-1);
    threadEnv[i].request = MAX_THRD;
  }

  return 0;
}
int init_graph() {
  int i;
  g_node = (Node*)malloc(sizeof(struct node_entry) * (g_num_nodes + 1));
  if (g_node == NULL)
    exit(-1);
  source = g_node;
  sink = g_node + g_num_nodes - 1;

  for (i = 0; i < g_num_nodes; i++) {
    g_node[i].height = 0;
    g_node[i].excess = 0;
    g_node[i].wave = 0;
    g_node[i].inQ = 0;
    g_node[i].adj_list = NULL;
  }
  source->height = g_num_nodes;
  source->inQ = 1;
  sink->inQ = 1;

  return 0;
}

int preflow(int n) {
  Edge edge;
  int i = 0;
  for (edge = g_node[0].adj_list; edge < g_node[1].adj_list; edge++) {
    g_node[0].excess -= edge->capacity;
    edge->endpoint->excess += (edge->capacity);
    edge->mateedge->capacity += (edge->capacity);
    edge->capacity = 0;
    if (edge->endpoint != sink && edge->endpoint->inQ == 0) {
      edge->endpoint->inQ = (i % n) + 1;
      enQ(threadEnv + i % n, edge->endpoint);
      i++;
    }
  }
}

//==============global relabel=========
int global_relabel(int tid) {
  // pthread_mutex_lock(&gr_mutex);
  // atomic_inc(&GRcnt);
  GRcnt++;
  int curWave = GRcnt;

  int* queue;
  //	int *curLevel;
  bool* color;
  queue = (int*)calloc(g_num_nodes, sizeof(int));
  //	curLevel = (int *)calloc(g_num_nodes, sizeof(int));
  color = (bool*)calloc(g_num_nodes, sizeof(bool));

  int depth;
  int level_end;
  int head, tail;
  int tmpn;
  int local_h;
  int isOK;
  Edge edge_f, edge_b;

  // initialize the queue
  queue[0] = g_num_nodes - 1;
  sink->wave = curWave;
  color[g_num_nodes - 1] = 1;
  color[0] = 1;
  head = 0;
  tail = 0;
  level_end = 0;

  depth = 1;

  // the bfs from sink
  while (head <= tail) {
    // depth = curLevel[queue[head]] +1;
    // pickup the head and find its neighbors
    for (edge_f = g_node[queue[head]].adj_list;
         edge_f < g_node[queue[head] + 1].adj_list;
         edge_f++) {
      edge_b = edge_f->mateedge;
      tmpn = edge_f->endpoint - g_node;
      if (color[tmpn] == 0 && edge_b->capacity > 0) {
        color[tmpn] = 1;
        tail++;
        queue[tail] = tmpn;

        /*
        isOK = -1;
        while (isOK == -1){
         local_h =  edge_f->endpoint->height;
         if( local_h < depth){
          //edge_f->endpoint->height = depth;
          isOK = cmpxchg(&(edge_f->endpoint->height), local_h,  depth);
          if (isOK != local_h){
          fprintf(stderr,"%d, %d!= %d\n", edge_f->endpoint-g_node, isOK,
        local_h);
          fflush(stderr);
            isOK =-1;
          }
        #ifdef GR_DEBUG
        fprintf(stderr, "grf %d -> %d\n", edge_f->endpoint-g_node, depth);
        fflush(stderr);
        #endif
         }else
           isOK = 0;

        }
    */
        pthread_mutex_lock(&(node_mutex[tmpn]));
        if (edge_f->endpoint->height < depth)
          edge_f->endpoint->height = depth;
        pthread_mutex_unlock(&(node_mutex[tmpn]));

        edge_f->endpoint->wave = curWave;
        // curLevel[tmpn] = depth;
      }
    }
    if (level_end == head) {
      depth++;
      level_end = tail;
    }
    head++;
  }

  source->wave = curWave;
#ifdef BACK_GR
  // if not all the nodes are covered, do a bfs from source
  if (tail < g_num_nodes - 2) {

    queue[0] = 0;
    head = 0;
    tail = 0;
    level_end = 0;
    //		curLevel[source-g_node] = g_num_nodes;

    depth = g_num_nodes + 1;
    while (head <= tail) {
      // depth ++;
      // pickup the head and find its neighbors
      for (edge_f = g_node[queue[head]].adj_list;
           edge_f < g_node[queue[head] + 1].adj_list;
           edge_f++) {
        edge_b = edge_f->mateedge;
        tmpn = edge_f->endpoint - g_node;
        if (color[tmpn] == 0 && edge_b->capacity > 0) {
          color[tmpn] = 1;
          tail++;
          queue[tail] = tmpn;

          /*
             isOK = -1;
             while (isOK == -1){
              local_h =  edge_f->endpoint->height;
              if( local_h < depth){
               //edge_f->endpoint->height = depth;
               isOK = cmpxchg(&(edge_f->endpoint->height), local_h,  depth);
               if (isOK != local_h){
               fprintf(stderr,"%d, %d!= %d\n", edge_f->endpoint-g_node, isOK,
             local_h);
               fflush(stderr);
                 isOK =-1;
               }
             #ifdef GR_DEBUG
             fprintf(stderr, "grf %d -> %d\n", edge_f->endpoint-g_node, depth);
             fflush(stderr);
             #endif
              }else
                isOK = 0;
             }
         */

          pthread_mutex_lock(&(node_mutex[tmpn]));
          if (edge_f->endpoint->height < depth) {
            edge_f->endpoint->height = depth;
#if defined(GR_DEBUG) || defined(DEBUG)
            fprintf(stderr, "grb %ld -> %d\n", edge_f->endpoint - g_node, depth);
            fflush(stderr);
#endif
          }
          pthread_mutex_unlock(&(node_mutex[tmpn]));
          edge_f->endpoint->wave = curWave;
        }
      }
      if (level_end == head) {
        depth++;
        level_end = tail;
      }
      head++;
    }
  }

#endif

  free(queue);
  free(color);

  return 0;
}

//==============global relabel=========

//==============Help=========
inline int send_work(int tid) {
  int j, k;
  int request_tid = threadEnv[tid].request;
  ThreadEnv *requester, *me;
  requester = threadEnv + threadEnv[tid].request;
  me = threadEnv + tid;

  k = 0;
  if (me->Qsize > MIN_Q_SIZE_TO_SHARE) {

    atomic_inc(&HPcnt);

    for (k = 0; k < SHARE_AMOUNT; k++) {
      j = me->tail - k - 1;
      if (j < 0)
        j += LOCAL_Q_SIZE;
      requester->Q[k] = me->Q[j];
      me->Q[j] = NoNode;
    }

    me->tail = j;
    me->Qsize -= k;
  }

  // accept new request
  me->request = MAX_THRD;
  // signal the requester
  requester->shareAmount = k;

  return 0;
}

inline int request_work(int tid) {
  int k, dest_thread, isReq;
  ThreadEnv *candidate, *me;

  if (flow_done())
    return 0;

  me = threadEnv + tid;

  // if anyone is waiting for me, let it go, because I don't have work to do
  // either
  if (me->request < MAX_THRD)
    threadEnv[me->request].shareAmount = 0;

  pthread_mutex_lock(&(thread_mutex[tid]));
  //printf("LOCK %d\n", tid);
  for (k = tid + 1; k < tid + num_threads; k++) {
    dest_thread = k % num_threads;
    candidate = threadEnv + dest_thread;

    if (pthread_mutex_trylock(&thread_mutex[dest_thread]) == 0) {
      if (candidate->request == MAX_THRD &&
          candidate->Qsize > MIN_Q_SIZE_TO_SHARE) {
        me->shareAmount = -1;
        candidate->request = tid;
        // waiting for other guy to share its queue
        while (me->shareAmount == -1) {
          if (flow_done()) {
            pthread_mutex_unlock(&(thread_mutex[dest_thread]));
            pthread_mutex_unlock(&(thread_mutex[tid]));
            return 1;
          }
        }

        if (me->shareAmount > 0) {
          pthread_mutex_unlock(&(thread_mutex[dest_thread]));
          // re-initializing thread
          me->head = 0;
          me->tail = me->shareAmount;
          me->Qsize = me->shareAmount;
          break;
        }
      }
      pthread_mutex_unlock(&(thread_mutex[dest_thread]));
    }
  } // for loop
  pthread_mutex_unlock(&(thread_mutex[tid]));
  //printf("UNLOCK %d\n", tid);
  return 0;
}


void* scan_nodes(void* threadid) {
  int i, j, k;
  long tid;

  int min_nbr_height;
  Edge nbr_edge, min_height_edge;
  long d;
  long local_e;
  long local_c;
  int max_flow;
  int isInQ;
  int height_before_lift;
  int node_push_times = 0, node_lift_times = 0;
  int op_times = 0;

  tid = (long)threadid;
  Node* cur_node;
  ThreadEnv* thisThread;
  thisThread = threadEnv + tid;

  if (tid == 0)
    global_relabel(tid);

  pthread_barrier_wait(&start_barrier);

  while (!flow_done()) {
    cur_node = thisThread->Q[0];

    while (cur_node != NoNode) {

#ifdef GR
      if (op_times > gr_threshold) {
        op_times = 0;
        if (pthread_mutex_trylock(&gr_mutex) == 0) {
          global_relabel(tid);
          pthread_mutex_unlock(&gr_mutex);
        }
      }
#endif

      while (cur_node->excess > 0) {
#ifdef DEBUG
          fprintf(stderr,
                  "%d h=%d, e=%ld\n",
                  cur_node - g_node,
                  cur_node->height,
                  cur_node->excess);
          fflush(stderr);
#endif
        min_nbr_height = 3 * g_num_nodes;
        for (nbr_edge = cur_node->adj_list; nbr_edge < (cur_node + 1)->adj_list;
             nbr_edge++) {

          if (nbr_edge->capacity > 0 &&
              (nbr_edge->endpoint)->height < min_nbr_height) {
            min_nbr_height = (nbr_edge->endpoint)->height;
            min_height_edge = nbr_edge;
          }
        }

#ifdef DEBUG
          fprintf(stderr, "work on %d\n", cur_node - g_node);
          fflush(stderr);
#endif
        if (cur_node->height > min_nbr_height) {
          local_e = cur_node->excess;
          local_c = min_height_edge->capacity;

          d = MIN(local_e, local_c);
          if (min_height_edge->endpoint->wave == cur_node->wave &&
              cur_node->height > min_height_edge->endpoint->height) {
            node_push_times++;
            op_times++;

            atomic_add(d, &(min_height_edge->mateedge->capacity));
            atomic_sub(d, &(min_height_edge->capacity));

            atomic_add(d, &((min_height_edge->endpoint)->excess));
            atomic_sub(d, &(cur_node->excess));

#if defined(PUSH) || defined(DEBUG)
              fprintf(stderr,
                      "[%ld] %ld(%ld) -> %ld -> %ld(%ld) \n",
                      tid,
                      cur_node - g_node,
                      cur_node->excess,
                      d,
                      min_height_edge->endpoint - g_node,
                      (min_height_edge->endpoint)->excess);
              fflush(stderr);
#endif
            // add min_nbr to local queue
            isInQ = cmpxchg(&(min_height_edge->endpoint->inQ), 0, tid + 1);
            if (isInQ == 0)
              enQ(thisThread, min_height_edge->endpoint);
          }
        } else {
          // if we cannot push to any nodes, then we must be able to lift
          node_lift_times++;
          op_times++;

          pthread_mutex_lock(&(node_mutex[cur_node - g_node]));
          if (cur_node->height < min_nbr_height + 1)
            cur_node->height = min_nbr_height + 1;
          pthread_mutex_unlock(&(node_mutex[cur_node - g_node]));
#if defined(LIFT) || defined(DEBUG)
            fprintf(stderr,
                    "%ld ^ %d, ref %ld(%d)\n",
                    cur_node - g_node,
                    cur_node->height,
                    min_height_edge->endpoint - g_node,
                    min_height_edge->endpoint->height);
            fflush(stderr);
#endif
        }
      }  // while( g_node[i].excess > 0 )
      set0(&(cur_node->inQ));

      if (cur_node->excess > 0) {
        isInQ = cmpxchg(&(cur_node->inQ), 0, tid + 1);
        if (isInQ == 0) {
          reenQ(thisThread, cur_node);
        } else {
          deQ(thisThread, cur_node);
        }
      } else {
        deQ(thisThread, cur_node);
      }

#ifdef HELP
      if (thisThread->request < MAX_THRD)
        send_work(tid);
#endif

      cur_node = thisThread->Q[thisThread->head];
    }  // while (i != -1)
#ifdef HELP
    // Q is empty, find something to do;
    request_work(tid);
#else
    break;
#endif
  }  // while(!flow_done())

  atomic_add(node_push_times, &(totalPushes));
  atomic_add(node_lift_times, &(totalLifts));
}  // scan_node

int main(long argc, char** argv) {
  int i, j;
  long t;
  FILE* fp;

  double duration;
  clock_t time;
  struct timeval tpstart, tpend;
  long iTimeInterval;

  int Test = 0;

  if (argc != 4) {
    printf("usage: %s .max_graph_file num_threads GR frequency\n", argv[0]);
    exit(-1);
  }
  num_threads = atoi(argv[2]);
  fprintf(stderr, "Threads: \t\t%d\n", num_threads);

  // initialize the graph, will read the graph from a file.
  fp = fopen(argv[1], "r");
  if (!fp) {
    printf("Cannot open %s\n", argv[1]);
    exit(-1);
  }

  char tag;
  char str[5];
  int from, to, capacity;

  fscanf(fp, "%c", &tag);
  while (tag != 'p') {
    while (getc(fp) != '\n')
      ;
    fscanf(fp, "%c", &tag);
  }
  fscanf(fp, "%s %d %d", str, &g_num_nodes, &g_num_edges);

  gr_threshold = (int)(atof(argv[3]) * g_num_nodes);
  fprintf(stderr, "GR Freq: \t\t%2.2f\n", atof(argv[3]));
  fprintf(stderr, "%d nodes, %d edges\n\n", g_num_nodes, g_num_edges);

  // initialize graph

  init_graph();

  // read and sort edges
  Edge edge = (Edge)malloc(2 * g_num_edges * sizeof(struct edge_entry));
  if (edge == NULL)
    exit(-1);
  edge_sym = edge;
  edgecnt = 0;
  for (i = 0; i < g_num_edges;) {
    if (getc(fp) == 'a') {
      fscanf(fp, "%d	%d	%d/n", &from, &to, &capacity);
      edge = Addedge(from - 1, to - 1, capacity, 0, edge);

      i++;
    }
  }
  assert(i == g_num_edges);

  SortEdges();
  fprintf(stderr, "Edge sorted!\n");

  // initialize thread parameters
  init_threads(num_threads);
  pthread_t threads[num_threads];

  // initial barrier and lock
  pthread_mutex_init(&(gr_mutex), NULL);
  node_mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t) * g_num_nodes);
  if (node_mutex == NULL)
    exit(-1);
  thread_mutex =
      (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t) * num_threads);
  if (thread_mutex == NULL)
    exit(-1);
  for (i = 0; i < num_threads; i++) {
    pthread_mutex_init(&(thread_mutex[i]), NULL);
  }
  for (i = 0; i < g_num_nodes; i++) {
    pthread_mutex_init(&(node_mutex[i]), NULL);
  }

  pthread_barrier_init(&start_barrier, NULL, num_threads);

  // now the f c e h arrays have been initialized. start the threads
  // so that they can work asynchronously.
  time = clock();
  gettimeofday(&tpstart, NULL);
  preflow(num_threads);
  for (t = 0; t < num_threads; t++) {
    pthread_create(&(threads[t]), NULL, scan_nodes, (void*)t);
  }
#ifdef MONITOR
  int num_idle_threads;
  int totalQsize, global_Q_size;
  while (!flow_done()){
    gettimeofday(&tpend, NULL);
    iTimeInterval = 1000000 * (tpend.tv_sec - tpstart.tv_sec);
    iTimeInterval += tpend.tv_usec - tpstart.tv_usec;
    duration = (double)iTimeInterval / 1000000;
    totalQsize = 0;
    num_idle_threads = 0;
    for (j = 0; j < num_threads; j++) {
      totalQsize += threadEnv[j].Qsize;
      if (threadEnv[j].Qsize == 0)
        num_idle_threads++;
    }
    if (duration > 10.0) {
                fprintf(stderr, " totalQsize: %8d globalQsize %4d idle %2d\n",
       totalQsize, global_Q_size, num_idle_threads);
              fflush(stderr);
                  exit(-1);
    }
    // printf("%12d %12d \t\t totalQsize: %8d GR %4d idel %2d\n",
    // g_node[0].excess,g_node[g_num_nodes-1].excess, totalQsize, GRcnt,
    // num_idle_threads);
    // printf("%d\n", totalQsize);
  }
  printf("\n");
#endif

  for (t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
  }
  gettimeofday(&tpend, NULL);
  time = clock() - time;

  iTimeInterval = 1000000 * (tpend.tv_sec - tpstart.tv_sec);
  iTimeInterval += tpend.tv_usec - tpstart.tv_usec;
  duration = (double)iTimeInterval / 1000000;

  fprintf(stderr, "FLOW: \t\t\t%ld\n", sink->excess);
  fprintf(stderr, "Wall Time: \t\t%5.3f\n", duration);
  fprintf(stderr,
          "CPU Time: \t\t%5.3f\n",
          (double)time / CLOCKS_PER_SEC / num_threads);
  fprintf(stderr, "Push: \t\t\t%ld\n", totalPushes);
  fprintf(stderr, "Push Efcy: \t\t%.1f\n", totalPushes / duration);
  fprintf(stderr, "Lift: \t\t\t%ld\n", totalLifts);
  fprintf(stderr, "Lift Efcy: \t\t%.1f\n", totalLifts / duration);
  fprintf(stderr, "GR: \t\t\t%d\n", GRcnt);
  fprintf(stderr, "HELP: \t\t\t%d\n", HPcnt);
  fflush(stderr);

  check_violation();

  pthread_exit(NULL);
}
