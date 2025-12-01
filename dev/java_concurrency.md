# Java Concurrency and Parallelism

## Table of Contents
1. [Core Synchronization Primitives](#core-synchronization-primitives)
2. [Thread Basics](#thread-basics)
3. [Core Interfaces](#core-interfaces)
4. [Executor Framework](#executor-framework)
5. [Synchronization Utilities](#synchronization-utilities)
6. [Lock API](#lock-api)
7. [Concurrent Collections](#concurrent-collections)
8. [Atomic Variables](#atomic-variables)
9. [CompletableFuture](#completablefuture)
10. [Fork/Join Framework](#forkjoin-framework)
11. [ThreadLocal](#threadlocal)
12. [Common Problems](#common-problems)
13. [Java Memory Model](#java-memory-model)
14. [Use Cases](#use-cases)

---

## Core Synchronization Primitives

### synchronized keyword
The `synchronized` keyword provides mutual exclusion using intrinsic locks (monitors). Every Java object has an associated monitor lock.

**Two forms:**
- **Synchronized method**: `public synchronized void method() { ... }` - locks on `this` (or class object for static methods)
- **Synchronized block**: `synchronized(object) { ... }` - locks on specified object

**Key points:**
- Only one thread can execute synchronized code on the same lock at a time
- Provides both mutual exclusion AND memory visibility guarantees
- Reentrant: same thread can acquire the lock multiple times
- Automatically releases lock on exception or normal exit

**Interview tip**: Prefer synchronized for simple cases; use explicit Locks for advanced features (try-lock, fairness, multiple conditions).

### volatile keyword
Declares that a variable's value will be modified by different threads. Provides visibility guarantees without full synchronization.

**Guarantees:**
- Reads/writes are atomic for all variables (including long/double)
- Changes by one thread are immediately visible to other threads
- Prevents instruction reordering around volatile variables (happens-before guarantee)

**Does NOT provide:**
- Compound operations atomicity (e.g., `counter++` is still not thread-safe)
- Mutual exclusion

**Common use cases:**
- Status flags: `private volatile boolean running = true;`
- Double-checked locking in singleton pattern
- Publishing immutable objects

**Interview tip**: Use volatile for simple flags; use AtomicXxx for counters/references requiring atomic updates.

---

## Thread Basics

### Thread class
Represents a thread of execution. Two ways to create: extend Thread or implement Runnable (preferred).

**Key methods:**
- `.start()` - starts thread execution (calls run() in new thread)
- `.run()` - contains the code to execute (don't call directly!)
- `.join()` - wait for this thread to die
- `.join(millis)` - wait at most millis milliseconds
- `.sleep(millis)` - static method, pauses current thread
- `.interrupt()` - interrupt this thread (sets interrupt flag)
- `.isInterrupted()` - test if thread is interrupted
- `.interrupted()` - static method, test and clear interrupt flag
- `.isAlive()` - test if thread is still running
- `.setPriority(int)` / `.getPriority()` - thread scheduling hint (1-10)
- `.setDaemon(boolean)` - mark as daemon thread (JVM exits when only daemon threads remain)
- `.getId()` / `.getName()` / `.setName()` - thread identification

**Thread states:**
1. **NEW** - created but not started
2. **RUNNABLE** - executing or ready to execute (includes OS-level running/ready)
3. **BLOCKED** - waiting to acquire a monitor lock
4. **WAITING** - waiting indefinitely (Object.wait(), Thread.join(), LockSupport.park())
5. **TIMED_WAITING** - waiting for specified time (Thread.sleep(), wait(timeout), join(timeout))
6. **TERMINATED** - completed execution

**Interview tip**: Always prefer ExecutorService over raw Threads for better resource management.

---

## Core Interfaces

### Runnable
Interface for tasks that can be executed concurrently but don't return a result.

```java
@FunctionalInterface
public interface Runnable {
    void run();
}
```

**Use when:** Fire-and-forget tasks, no result needed, no checked exceptions to throw.

### Callable<V>
Like Runnable but returns a result and can throw checked exceptions.

```java
@FunctionalInterface
public interface Callable<V> {
    V call() throws Exception;
}
```

**Use when:** Task needs to return a result or throw checked exceptions.

### Future<V>
Represents the result of an asynchronous computation. Obtained from ExecutorService.submit().

**Key methods:**
- `.get()` - blocks until result is available, throws ExecutionException if task failed
- `.get(timeout, unit)` - blocks with timeout, throws TimeoutException
- `.cancel(mayInterruptIfRunning)` - attempt to cancel execution
- `.isDone()` - returns true if completed (normally, exceptionally, or cancelled)
- `.isCancelled()` - returns true if cancelled before completion

**Interview tip**: Future is blocking; for non-blocking async, use CompletableFuture.

---

## Executor Framework

### Executor
Minimal interface that decouples task submission from execution mechanics.

```java
public interface Executor {
    void execute(Runnable command);
}
```

**Use when:** You only need to execute Runnables without lifecycle management.

### ExecutorService
Extends Executor, adds lifecycle management and ability to produce Futures.

**Key methods:**
- `.submit(Callable<T>)` - returns Future<T>
- `.submit(Runnable)` - returns Future<?> (result is null)
- `.submit(Runnable, T result)` - returns Future<T> with specified result
- `.invokeAll(Collection<Callable<T>>)` - executes all, returns List<Future<T>>
- `.invokeAny(Collection<Callable<T>>)` - returns result of one that succeeds
- `.shutdown()` - initiates orderly shutdown (no new tasks, completes existing)
- `.shutdownNow()` - attempts to stop all executing tasks, returns list of waiting tasks
- `.awaitTermination(timeout, unit)` - blocks until all tasks complete after shutdown
- `.isShutdown()` / `.isTerminated()` - status checks

**Interview tip**: Always shutdown ExecutorService (use try-with-resources if using ExecutorService from Java 19+).

### ScheduledExecutorService
Extends ExecutorService for delayed or periodic task execution.

**Key methods:**
- `.schedule(Callable/Runnable, delay, unit)` - execute once after delay
- `.scheduleAtFixedRate(Runnable, initialDelay, period, unit)` - execute periodically at fixed rate (regardless of execution time)
- `.scheduleWithFixedDelay(Runnable, initialDelay, delay, unit)` - execute periodically with fixed delay between completion and next start

**Interview tip:** Fixed rate = start times are fixed; Fixed delay = delay between executions is fixed.

### ThreadPoolExecutor
Main implementation of ExecutorService, manages a pool of worker threads.

**Constructor parameters:**
- `corePoolSize` - minimum number of threads to keep alive
- `maximumPoolSize` - maximum number of threads
- `keepAliveTime` - time to keep idle threads alive (above core size)
- `workQueue` - queue to hold tasks before execution
- `threadFactory` - factory to create new threads
- `rejectedExecutionHandler` - handler for rejected tasks

**Queue types affect behavior:**
- **Direct handoff** (SynchronousQueue): no queuing, create thread immediately or reject
- **Unbounded queue** (LinkedBlockingQueue): unlimited queue, never reaches maxPoolSize
- **Bounded queue** (ArrayBlockingQueue): limited queue, creates threads up to max, then rejects

**Key additional methods:**
- `.getPoolSize()` - current number of threads
- `.getActiveCount()` - approximate number of actively executing tasks
- `.getCompletedTaskCount()` - approximate total number of completed tasks
- `.setCorePoolSize()` / `.setMaximumPoolSize()` - dynamic resizing

**Interview tip:** Understanding ThreadPoolExecutor sizing and queue behavior is crucial.

### ForkJoinPool
Special ExecutorService designed for divide-and-conquer parallelism using work-stealing algorithm.

**Key features:**
- Work-stealing: idle threads steal work from busy threads' deques
- Best for recursive divide-and-conquer tasks
- Default parallelism = number of CPU cores

**Key methods:**
- `.invoke(ForkJoinTask)` - execute and wait for result
- `.submit(ForkJoinTask)` - async execution
- `.commonPool()` - static method, returns shared pool (used by parallel streams)

**Interview tip:** ForkJoinPool powers Java parallel streams and CompletableFuture async methods.

### Executors Factory Methods

**Executors.newFixedThreadPool(n)**
- Creates ThreadPoolExecutor with fixed number of threads
- Uses unbounded LinkedBlockingQueue
- All n threads are kept alive (even if idle)
- **Use when:** Known load, need to limit concurrent threads

**Executors.newCachedThreadPool()**
- Creates ThreadPoolExecutor with 0 core, unbounded max threads
- Uses SynchronousQueue (direct handoff)
- Threads idle for 60s are terminated
- **Use when:** Many short-lived async tasks
- **Warning:** Can create unlimited threads, risk OutOfMemoryError

**Executors.newSingleThreadExecutor()**
- Single worker thread with unbounded queue
- Guarantees sequential execution
- **Use when:** Need serialized execution, maintain order

**Executors.newWorkStealingPool()**
- Creates ForkJoinPool with parallelism = CPU cores
- **Use when:** Divide-and-conquer algorithms, parallel processing

**Executors.newThreadPerTaskExecutor()**
- Creates new platform thread for each task (Java 19+)
- No pooling
- **Use when:** Isolation needed, debugging

**Executors.newVirtualThreadPerTaskExecutor()**
- Creates new virtual thread for each task (Java 21+)
- Lightweight, can create millions
- **Use when:** High-concurrency I/O-bound tasks

**Executors.newScheduledThreadPool(n)**
- Creates ScheduledThreadPoolExecutor with n core threads
- **Use when:** Need multiple threads for scheduled tasks

**Executors.newSingleThreadScheduledExecutor()**
- Single thread for scheduled tasks
- **Use when:** Serial scheduled execution

**Executors.unconfigurableExecutorService(ExecutorService)**
- Wraps ExecutorService to prevent reconfiguration
- **Use when:** Expose service without allowing parameter changes

**Executors.defaultThreadFactory() / privilegedThreadFactory()**
- Factory methods for creating threads with default or privileged access control context

**Interview tip:** In production, prefer custom ThreadPoolExecutor over convenience methods for better control and monitoring.

---

## Synchronization Utilities

### CountDownLatch
One-time synchronization barrier. Threads wait until latch counts down to zero.

**Constructor:** `CountDownLatch(int count)`

**Key methods:**
- `.await()` - blocks until count reaches zero
- `.await(timeout, unit)` - blocks with timeout
- `.countDown()` - decrements count
- `.getCount()` - current count

**Use cases:**
- Wait for N tasks to complete before proceeding
- Ensure service dependencies are ready before starting
- Coordinating test threads

**Example pattern:**
```java
CountDownLatch startSignal = new CountDownLatch(1);
CountDownLatch doneSignal = new CountDownLatch(N);
// Workers wait on startSignal, decrement doneSignal when done
// Main decrements startSignal to start all, waits on doneSignal
```

**Interview tip:** One-time use only (can't reset). For reusable barrier, use CyclicBarrier.

### CyclicBarrier
Reusable synchronization barrier where threads wait until all parties arrive.

**Constructor:** `CyclicBarrier(int parties)` or `CyclicBarrier(int parties, Runnable barrierAction)`

**Key methods:**
- `.await()` - wait until all parties invoke await
- `.await(timeout, unit)` - wait with timeout
- `.reset()` - reset barrier to initial state
- `.getNumberWaiting()` / `.getParties()` - status info
- `.isBroken()` - check if barrier is broken (due to interruption/timeout)

**Use cases:**
- Parallel algorithms with phases (all threads complete phase before proceeding)
- Simulations where entities must synchronize at steps

**Interview tip:** Difference from CountDownLatch: CyclicBarrier is reusable and threads wait for each other (mutual waiting), while CountDownLatch is one-way (waiters wait for counters).

### Semaphore
Controls access to shared resources through permits. Like a generalized lock (allows N concurrent accesses).

**Constructor:** `Semaphore(int permits)` or `Semaphore(int permits, boolean fair)`

**Key methods:**
- `.acquire()` - acquire one permit (blocks if unavailable)
- `.acquire(n)` - acquire n permits
- `.tryAcquire()` - try to acquire without blocking
- `.tryAcquire(timeout, unit)` - try with timeout
- `.release()` - release permit
- `.release(n)` - release n permits
- `.availablePermits()` - current available permits

**Use cases:**
- Resource pools (connection pools, limited API rate)
- Throttling concurrent access
- Implementing bounded buffers

**Interview tip:** Semaphore(1) = binary semaphore (like mutex), but not reentrant and not tied to thread ownership.

### Phaser
More flexible synchronization barrier supporting dynamic parties and multiple phases.

**Key features:**
- Dynamic registration/deregistration
- Multi-phase synchronization
- Can terminate after phase
- Tree-structured for scalability

**Key methods:**
- `.register()` - add new party
- `.bulkRegister(int)` - add multiple parties
- `.arriveAndAwaitAdvance()` - arrive and wait for others (like CyclicBarrier.await())
- `.arrive()` - arrive but don't wait
- `.arriveAndDeregister()` - arrive and deregister
- `.awaitAdvance(int phase)` - wait for specific phase
- `.getPhase()` - current phase number
- `.isTerminated()` - check if phaser is terminated

**Use cases:**
- Complex multi-phase algorithms
- Dynamic number of participants
- Tasks that complete in waves

**Interview tip:** More complex than CyclicBarrier but more powerful for advanced coordination patterns.

### Exchanger<V>
Synchronization point where two threads can exchange objects.

**Key method:**
- `.exchange(V x)` - wait for another thread, swap objects
- `.exchange(V x, timeout, unit)` - exchange with timeout

**Use cases:**
- Pipeline designs (thread A produces, thread B consumes, swap buffers)
- Genetic algorithms (swap chromosomes between threads)
- Bidirectional data transfer

**Interview tip:** Only works with pairs of threads. Rarely used but good to know conceptually.

---

## Lock API

More flexible than synchronized, part of `java.util.concurrent.locks` package.

### Lock Interface
Base interface for explicit locks.

**Key methods:**
- `.lock()` - acquire lock (blocks if unavailable)
- `.unlock()` - release lock (must be called in finally block!)
- `.tryLock()` - try to acquire without blocking
- `.tryLock(timeout, unit)` - try with timeout
- `.newCondition()` - create Condition object for this lock
- `.lockInterruptibly()` - acquire lock unless interrupted

**Basic pattern:**
```java
Lock lock = new ReentrantLock();
lock.lock();
try {
    // critical section
} finally {
    lock.unlock(); // always in finally!
}
```

### ReentrantLock
Main Lock implementation, similar to synchronized but with more features.

**Features:**
- Reentrant (like synchronized)
- Optional fairness (fair locks favor longest-waiting thread)
- Try-lock capabilities
- Interruptible lock acquisition
- Can have multiple Condition objects

**Constructor:** `ReentrantLock()` or `ReentrantLock(boolean fair)`

**Additional methods:**
- `.isHeldByCurrentThread()` - check if current thread holds lock
- `.getHoldCount()` - number of holds by current thread
- `.isLocked()` - check if any thread holds lock
- `.hasQueuedThreads()` - check if threads are waiting
- `.getQueueLength()` - estimate of waiting threads

**When to use over synchronized:**
- Need try-lock or timed-lock
- Need interruptible lock acquisition
- Need fairness guarantee
- Need multiple condition variables
- Want hand-over-hand locking in data structures

### ReadWriteLock Interface
Maintains a pair of locks: one for read-only operations, one for writing.

**Key idea:** Multiple readers can hold the lock simultaneously, but writers need exclusive access.

**Key methods:**
- `.readLock()` - returns Lock for reading
- `.writeLock()` - returns Lock for writing

### ReentrantReadWriteLock
Main ReadWriteLock implementation.

**Features:**
- Multiple concurrent readers OR one exclusive writer
- Optional fairness
- Write lock can be downgraded to read lock
- Reentrant for both read and write locks

**Use cases:**
- Read-heavy data structures (caches, registries)
- When reads are much more frequent than writes

**Interview tip:** Read locks cannot upgrade to write locks directly (would deadlock). Must release read, acquire write.

### StampedLock
Advanced lock with optimistic reading (Java 8+).

**Three modes:**
1. **Writing** - exclusive lock
2. **Reading** - non-exclusive lock
3. **Optimistic reading** - no actual locking, validate later

**Key methods:**
- `.writeLock()` / `.unlockWrite(stamp)`
- `.readLock()` / `.unlockRead(stamp)`
- `.tryOptimisticRead()` - returns stamp (0 if write-locked)
- `.validate(stamp)` - check if stamp still valid

**Advantages over ReentrantReadWriteLock:**
- Optimistic reads have no contention
- Lock views can be converted (upgrade/downgrade)
- Better performance for read-heavy scenarios

**Disadvantages:**
- Not reentrant
- No condition variables
- More complex to use

**Interview tip:** StampedLock is often faster than ReentrantReadWriteLock but harder to use correctly.

### Condition Interface
Provides await/signal mechanism similar to Object.wait/notify but more flexible.

**Key methods:**
- `.await()` - wait until signaled (like Object.wait())
- `.await(timeout, unit)` - wait with timeout
- `.signal()` - wake up one waiting thread (like notify())
- `.signalAll()` - wake up all waiting threads (like notifyAll())

**Advantages over wait/notify:**
- Multiple conditions per lock
- More readable code
- Interruptible and timed waits with better exception handling

**Pattern:**
```java
Lock lock = new ReentrantLock();
Condition notEmpty = lock.newCondition();
Condition notFull = lock.newCondition();

// Producer
lock.lock();
try {
    while (isFull()) notFull.await();
    // produce
    notEmpty.signal();
} finally { lock.unlock(); }

// Consumer
lock.lock();
try {
    while (isEmpty()) notEmpty.await();
    // consume
    notFull.signal();
} finally { lock.unlock(); }
```

---

## Concurrent Collections

Thread-safe collections optimized for concurrent access. Better performance than Collections.synchronizedXxx() wrappers.

### ConcurrentHashMap<K,V>
Thread-safe hash table with fine-grained locking (segment-based in Java 7, lock-free in Java 8+).

**Features:**
- No locking on reads (mostly)
- Fine-grained locking on writes
- Iterators are weakly consistent (don't throw ConcurrentModificationException)
- Null keys/values not allowed
- Atomic compound operations

**Key methods (beyond Map):**
- `.putIfAbsent(k, v)` - atomic put if absent
- `.remove(k, v)` - atomic remove if value matches
- `.replace(k, v)` - atomic replace
- `.replace(k, oldV, newV)` - atomic replace if old value matches
- `.computeIfAbsent(k, mappingFunction)` - atomic compute if absent (Java 8+)
- `.computeIfPresent(k, remappingFunction)` - atomic compute if present
- `.compute(k, remappingFunction)` - atomic compute
- `.merge(k, v, remappingFunction)` - atomic merge

**Use cases:**
- Caches
- Shared mutable state
- High-concurrency maps

**Interview tip:** Preferred over Hashtable and Collections.synchronizedMap. Size may be approximate.

### ConcurrentLinkedQueue<E>
Unbounded thread-safe FIFO queue based on linked nodes. Lock-free using CAS operations.

**Features:**
- Non-blocking algorithm
- Unbounded (limited by memory)
- Weakly consistent iterators
- Null elements not allowed

**Use cases:**
- Producer-consumer with multiple producers/consumers
- Task queues

### ConcurrentLinkedDeque<E>
Unbounded thread-safe double-ended queue. Lock-free using CAS.

**Features:**
- Can add/remove from both ends
- Non-blocking
- Weakly consistent iterators

**Use cases:**
- Work-stealing queues
- Dual-ended access patterns

### BlockingQueue<E> Interface
Queue that supports operations that wait for queue to become non-empty (take) or non-full (put).

**Key blocking methods:**
- `.put(e)` - insert, wait if full
- `.take()` - retrieve and remove, wait if empty
- `.offer(e, timeout, unit)` - insert with timeout
- `.poll(timeout, unit)` - retrieve with timeout

**Key non-blocking methods:**
- `.offer(e)` - insert if possible, return false if full
- `.poll()` - retrieve if available, return null if empty
- `.remainingCapacity()` - remaining capacity

**Implementations:**

#### LinkedBlockingQueue<E>
- Optionally bounded linked queue
- FIFO ordering
- Separate locks for put/take (better concurrency)
- Default capacity: Integer.MAX_VALUE

**Use when:** High throughput, don't need bounded capacity

#### ArrayBlockingQueue<E>
- Bounded array-based queue
- FIFO ordering
- Single lock for put/take
- Optional fairness for blocking operations

**Use when:** Need bounded queue, fixed capacity

#### PriorityBlockingQueue<E>
- Unbounded priority queue
- Elements ordered by natural order or Comparator
- Takes are blocked when empty, puts never block

**Use when:** Priority-based processing

#### DelayQueue<E extends Delayed>
- Unbounded queue of Delayed elements
- Element can only be taken when its delay expires
- Head is element that expired longest ago

**Use when:** Scheduled task execution, cache expiration

#### SynchronousQueue<E>
- Queue with no internal capacity
- Each put must wait for take (and vice versa)
- Useful for direct handoff between threads

**Use when:** Direct thread-to-thread handoff, no buffering

**Interview tip:** BlockingQueue is the standard way to implement producer-consumer patterns.

### CopyOnWriteArrayList<E>
Thread-safe List where all mutative operations create a fresh copy of underlying array.

**Features:**
- Reads never block (no synchronization)
- Writes are expensive (full copy)
- Iterators use snapshot, never throw ConcurrentModificationException
- Iteration is much more frequent than mutation

**Use cases:**
- Observer/listener lists
- Mostly read, rarely modified collections

**Trade-off:** Fast reads, slow writes + memory overhead

### CopyOnWriteArraySet<E>
Thread-safe Set backed by CopyOnWriteArrayList.

**Features/use cases:** Same as CopyOnWriteArrayList

**Interview tip:** For small sets that are read frequently but modified rarely.

---

## Atomic Variables

Classes in `java.util.concurrent.atomic` that support lock-free, thread-safe operations on single variables using Compare-And-Swap (CAS) hardware primitives.

### AtomicInteger / AtomicLong
Thread-safe integers/longs with atomic operations.

**Key methods:**
- `.get()` / `.set(newValue)` - simple read/write
- `.getAndSet(newValue)` - atomic swap
- `.compareAndSet(expect, update)` - atomic CAS
- `.getAndIncrement()` / `.incrementAndGet()` - atomic i++ / ++i
- `.getAndDecrement()` / `.decrementAndGet()` - atomic i-- / --i
- `.getAndAdd(delta)` / `.addAndGet(delta)` - atomic addition
- `.updateAndGet(IntUnaryOperator)` - atomic custom update (Java 8+)
- `.getAndUpdate(IntUnaryOperator)` - atomic custom update
- `.accumulateAndGet(x, IntBinaryOperator)` - atomic accumulation

**Use cases:**
- Counters, sequence numbers
- Lock-free algorithms

### AtomicBoolean
Thread-safe boolean with atomic operations.

**Key methods:**
- `.get()` / `.set(newValue)`
- `.compareAndSet(expect, update)` - atomic CAS
- `.getAndSet(newValue)` - atomic swap

**Use cases:**
- Flags, one-time initialization guards

### AtomicReference<V>
Thread-safe reference with atomic operations.

**Key methods:**
- `.get()` / `.set(newValue)`
- `.compareAndSet(expect, update)` - atomic CAS (uses object equality)
- `.getAndSet(newValue)` - atomic swap
- `.updateAndGet(UnaryOperator)` - atomic custom update
- `.getAndUpdate(UnaryOperator)` - atomic custom update

**Use cases:**
- Lock-free data structures
- Immutable object updates

### AtomicReferenceArray<E>
Array of atomic references.

### AtomicIntegerArray / AtomicLongArray
Arrays of atomic integers/longs.

### AtomicStampedReference<V>
AtomicReference with integer "stamp" to solve ABA problem.

**Key methods:**
- `.compareAndSet(expectedRef, newRef, expectedStamp, newStamp)`
- `.get(int[] stampHolder)` - get reference and stamp

**Use cases:**
- Lock-free algorithms that need to detect if value was changed and changed back (ABA problem)

### AtomicMarkableReference<V>
AtomicReference with boolean "mark".

**Interview tip:** Atomic classes are building blocks for lock-free algorithms. Much faster than locks for simple operations but harder to use correctly for complex updates.

---

## CompletableFuture

Powerful API for asynchronous programming (Java 8+). Combines Future with composable async operations.

**Key features:**
- Manual completion
- Chaining/composition of async operations
- Exception handling
- Combining multiple futures

**Creating CompletableFuture:**
```java
// From value
CompletableFuture<String> completed = CompletableFuture.completedFuture("result");

// Async with default executor (ForkJoinPool.commonPool)
CompletableFuture<String> cf = CompletableFuture.supplyAsync(() -> "result");

// Async with custom executor
CompletableFuture<String> cf = CompletableFuture.supplyAsync(() -> "result", executor);

// Manual completion
CompletableFuture<String> cf = new CompletableFuture<>();
cf.complete("result"); // or cf.completeExceptionally(exception);
```

**Key transformation methods:**
- `.thenApply(Function)` - transform result (like map)
- `.thenApplyAsync(Function)` - transform in different thread
- `.thenAccept(Consumer)` - consume result (void)
- `.thenRun(Runnable)` - run action after completion
- `.thenCompose(Function<T, CompletableFuture<U>>)` - flat map

**Key combination methods:**
- `.thenCombine(other, BiFunction)` - combine two futures
- `.thenAcceptBoth(other, BiConsumer)` - consume both results
- `.runAfterBoth(other, Runnable)` - run after both complete
- `.applyToEither(other, Function)` - use whichever completes first
- `.allOf(CompletableFuture<?>...)` - static, complete when all complete
- `.anyOf(CompletableFuture<?>...)` - static, complete when any completes

**Exception handling:**
- `.exceptionally(Function)` - handle exception, provide fallback
- `.handle(BiFunction)` - handle both result and exception
- `.whenComplete(BiConsumer)` - side effect on completion (doesn't transform)

**Example:**
```java
CompletableFuture<String> future = CompletableFuture
    .supplyAsync(() -> fetchFromDB())
    .thenApply(data -> transform(data))
    .thenApply(transformed -> validate(transformed))
    .exceptionally(ex -> "default")
    .thenAccept(result -> save(result));
```

**Interview tip:** CompletableFuture is the modern way to do async in Java. Know the difference between `thenApply` (sync in same thread) and `thenApplyAsync` (async in executor).

---

## Fork/Join Framework

Framework for parallel divide-and-conquer algorithms using work-stealing.

### ForkJoinTask<V>
Abstract base class for tasks that run in ForkJoinPool.

**Key methods:**
- `.fork()` - arrange async execution in current pool
- `.join()` - wait for completion and return result
- `.invoke()` - fork and join together (convenience)
- `.get()` - like join but checks exceptions

**Subclasses:**

### RecursiveTask<V>
ForkJoinTask that returns a result.

**Pattern:**
```java
class SumTask extends RecursiveTask<Long> {
    private static final int THRESHOLD = 1000;
    private long[] array;
    private int start, end;

    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // Base case: compute directly
            long sum = 0;
            for (int i = start; i < end; i++) sum += array[i];
            return sum;
        } else {
            // Recursive case: split
            int mid = (start + end) / 2;
            SumTask left = new SumTask(array, start, mid);
            SumTask right = new SumTask(array, mid, end);
            left.fork(); // async execute left
            long rightResult = right.compute(); // compute right in current thread
            long leftResult = left.join(); // wait for left
            return leftResult + rightResult;
        }
    }
}
```

### RecursiveAction
ForkJoinTask that doesn't return a result (like Runnable).

**Use cases:**
- Array sorting (parallel merge sort, quick sort)
- Tree traversal
- Matrix operations
- Any divide-and-conquer algorithm

**Interview tip:** Fork/Join is automatic in parallel streams. The threshold for splitting is crucial for performance.

---

## ThreadLocal

Provides thread-local variables. Each thread accessing such variable has its own, independently initialized copy.

**Key methods:**
- `.get()` - get current thread's value
- `.set(T value)` - set current thread's value
- `.remove()` - remove current thread's value (important to prevent leaks!)
- `.withInitial(Supplier)` - static factory with initial value supplier (Java 8+)

**Usage:**
```java
private static final ThreadLocal<DateFormat> formatter =
    ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyy-MM-dd"));

// Each thread gets its own DateFormat instance
DateFormat df = formatter.get();
```

**Use cases:**
- Per-thread context (user ID, transaction ID)
- Non-thread-safe objects (SimpleDateFormat, database connections)
- Performance optimization (avoid object creation per call)

**Dangers:**
- Memory leaks in thread pools (threads reused, ThreadLocal not cleared)
- Always call `.remove()` in finally block or use try-with-resources pattern

**Interview tip:** ThreadLocal + thread pools = potential memory leak. Always clean up!

---

## Common Problems

### Race Condition
Multiple threads access shared data and outcome depends on execution order.

**Solution:** Synchronization, atomic variables, or thread-safe collections.

### Deadlock
Two or more threads are blocked forever, each waiting for the other to release a lock.

**Classic example:**
```java
// Thread 1
synchronized(lock1) {
    synchronized(lock2) { ... }
}

// Thread 2
synchronized(lock2) {
    synchronized(lock1) { ... } // DEADLOCK!
}
```

**Prevention strategies:**
- Lock ordering: always acquire locks in same order
- Lock timeout: use tryLock with timeout
- Deadlock detection: monitor thread dumps
- Avoid nested locks

### Livelock
Threads aren't blocked but continuously change states in response to each other without making progress.

**Example:** Two threads keep yielding to each other.

**Solution:** Add randomness to retry logic, backoff strategies.

### Thread Starvation
A thread cannot gain access to shared resources and cannot make progress.

**Causes:** Low thread priority, unfair locks, poor scheduler

**Solution:** Fair locks, proper thread priorities, avoid long-running operations in critical sections.

### ABA Problem
In lock-free algorithms, a value changes from A to B and back to A, but CAS doesn't detect the change.

**Solution:** AtomicStampedReference or AtomicMarkableReference.

---

## Java Memory Model

Understanding happens-before relationships and memory visibility.

### Happens-Before Relationship
Defines when actions in one thread are guaranteed to be visible to another thread.

**Key happens-before rules:**
1. **Program order rule**: Each action in a thread happens-before every subsequent action in the same thread
2. **Monitor lock rule**: Unlock on a monitor happens-before every subsequent lock on the same monitor
3. **Volatile variable rule**: Write to volatile field happens-before every subsequent read of that field
4. **Thread start rule**: Thread.start() happens-before any action in the started thread
5. **Thread join rule**: Actions in a thread happen-before Thread.join() returns in another thread
6. **Transitivity**: If A happens-before B, and B happens-before C, then A happens-before C

### Memory Visibility
Without proper synchronization, threads may see stale values due to CPU caches and compiler optimizations.

**Mechanisms ensuring visibility:**
- volatile variables
- synchronized blocks
- Lock.lock()/unlock()
- Thread start/join
- Atomic operations
- Final fields (published safely)

**Interview tip:** "Happens-before" is the key concept. Know that synchronization provides both mutual exclusion AND memory visibility.

---

## Use Cases

### Running async tasks and getting results
→ **ExecutorService** with **Callable** and **Future** (or **CompletableFuture** for modern async)

### Scheduling tasks (delayed or periodic)
→ **ScheduledExecutorService** (`.schedule()`, `.scheduleAtFixedRate()`, `.scheduleWithFixedDelay()`)

### Waiting for multiple threads to complete
→ **CountDownLatch** (one-time barrier, waiters wait for counters)

### Synchronizing threads at a barrier
→ **CyclicBarrier** (reusable, mutual waiting) or **Phaser** (more complex multi-phase scenarios)

### Limiting concurrent access to resource
→ **Semaphore** (permits-based access control)

### Thread-safe collections for concurrent access
→ **ConcurrentHashMap** (maps), **CopyOnWriteArrayList** (read-heavy lists), **ConcurrentLinkedQueue** (queues)

### Producer-consumer pattern
→ **BlockingQueue** implementations (**LinkedBlockingQueue** or **ArrayBlockingQueue**)

### Atomic operations without locks
→ **AtomicInteger**, **AtomicLong**, **AtomicReference**, etc.

### Parallel divide-and-conquer algorithms
→ **ForkJoinPool** with **RecursiveTask**/**RecursiveAction**

### Complex async pipelines and composition
→ **CompletableFuture** with chaining methods

### Read-heavy, write-rare scenarios
→ **ReentrantReadWriteLock** or **StampedLock** (optimistic reads)

### Per-thread state/context
→ **ThreadLocal**

### Fairness requirements
→ **Semaphore(permits, true)** or **ReentrantLock(true)**

### Need try-lock or timed-lock
→ **ReentrantLock** with `.tryLock()` or `.tryLock(timeout, unit)`

### High-concurrency I/O-bound tasks
→ **Virtual threads** (Java 21+) with **newVirtualThreadPerTaskExecutor()**

---

## Interview Preparation Tips

1. **Know when to use what**: Understand trade-offs between synchronized vs Lock, Future vs CompletableFuture, different ExecutorService types

2. **Understand internals**: How ConcurrentHashMap works, how ForkJoinPool does work-stealing, how AtomicXxx uses CAS

3. **Memory model basics**: Happens-before, visibility, volatile guarantees

4. **Common patterns**: Producer-consumer, fork-join, async pipelines, double-checked locking

5. **Pitfalls**: ThreadLocal leaks, forgetting to shutdown ExecutorService, deadlocks, race conditions

6. **Modern Java**: Virtual threads (Java 21), StructuredConcurrency (preview), ScopedValue

7. **Practical coding**: Be ready to implement simple concurrent solutions using ExecutorService, CompletableFuture, or BlockingQueue

**Most important for interviews:**
- ExecutorService and thread pools
- CompletableFuture
- ConcurrentHashMap
- BlockingQueue
- synchronized vs ReentrantLock
- volatile
- Atomic classes
- Basic Fork/Join understanding
