#define __xg(x) ((volatile long *)(x))

#define atomic_set(v,i)    (((v)->counter) = (i))

typedef struct { volatile long counter; } atomic_t;

static inline unsigned long __cmpxchg(volatile void *ptr, long old,
                 long new, int size)
{

   unsigned long prev;
   switch (size) {
   case 1:
      __asm__ __volatile__("lock; cmpxchgb %b1,%2"
                 : "=a"(prev)
                 : "q"(new), "m"(*__xg(ptr)), "0"(old)
                 : "memory");
      return prev;
   case 2:
      __asm__ __volatile__("lock; cmpxchgw %w1,%2"
                 : "=a"(prev)
                 : "r"(new), "m"(*__xg(ptr)), "0"(old)
                 : "memory");
      return prev;
   case 4:
      __asm__ __volatile__("lock; cmpxchgl %k1,%2"
                 : "=a"(prev)
                 : "r"(new), "m"(*__xg(ptr)), "0"(old)
                 : "memory");
      return prev;
   case 8:
      __asm__ __volatile__("lock; cmpxchgq %1,%2"
                 : "=a"(prev)
                 : "r"(new), "m"(*__xg(ptr)), "0"(old)
                 : "memory");
      return prev;
   }
   return old;
}
static __inline__ void atomic_inc(volatile int *v)
{
        __asm__ __volatile__(
               "lock; incl %0"
                :"=m" (*v)
                :"m" (*v));
}

static __inline__ void atomic_dec(volatile int *v)
{
        __asm__ __volatile__(
                "lock; decl %0"
                :"=m" (*v)
                :"m" (*v));
}



static __inline__ void atomic_sub(int i, volatile long *v)
{
        __asm__ __volatile__(
                "lock; subl %1,%0"
                :"=m" (*v)
                :"ir" (i), "m" (*v));
}
static __inline__ void atomic_add(int i, volatile long *v)
{
        __asm__ __volatile__(
                "lock; addl %1,%0"
                :"=m" (*v)
                :"ir" (i), "m" (*v));
}


static inline unsigned long __xchg(long x, volatile void * ptr, int size)
{
	//printf("%d\n", size);
        switch (size) {
                case 1:
                        __asm__ __volatile__("xchgb %b0,%1"
                                :"=q" (x)
                                :"m" (*__xg(ptr)), "0" (x)
                                :"memory");
                        break;
                case 2:
                        __asm__ __volatile__("xchgw %w0,%1"
                                :"=r" (x)
                                :"m" (*__xg(ptr)), "0" (x)
                                :"memory");
                        break;
                case 4:
                        __asm__ __volatile__("xchgl %k0,%1"
                                :"=r" (x)
                                :"m" (*__xg(ptr)), "0" (x)
                                :"memory");
                        break;
                case 8:
                        __asm__ __volatile__("xchgq %0,%1"
                                :"=r" (x)
                                :"m" (*__xg(ptr)), "0" (x)
                                :"memory");
                        break;
        }
        return x;
}
