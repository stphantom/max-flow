#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) < (y) ? (y) : (x))

//#define DEBUG

//#define GR_DEBUG
//#define LIFT
//#define PUSH
//#define MONITOR

#define GR
#define BACK_GR
#define HELP

#define MAX_THRD 1000 // max number of threads

#define MIN_Q_SIZE_TO_SHARE 200
#define SHARE_AMOUNT (MIN_Q_SIZE_TO_SHARE/2)
#define LOCAL_Q_SIZE g_num_nodes

#define xchg(ptr, v) \
  ((__typeof__(*(ptr)))__xchg((unsigned long)(v), (ptr), sizeof(*(ptr))))
#define cmpxchg(ptr, o, n)        \
  ((__typeof__(*(ptr)))__cmpxchg( \
      (ptr), (unsigned long)(o), (unsigned long)(n), sizeof(*(ptr))))
#define set0(ptr) (xchg((ptr), 0))
