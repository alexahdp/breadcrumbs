# Java.util.concurrent

## Interfaces and classes for concurrent programming in Java.
**Callable** - interface that represents a task that returns a result and may throw an exception. Key method: `.call()`.
**Runnable** - interace that represents a task that can be executed concurrently. Key method: `.run()`.
**Future** - interface that represents the result of an asynchronous computation. Key methods: `.get()`, `.cancel()`, `.isDone()`, `.isCancelled()`.


**Executor.execute(Runnable)** - a minimal lightweight interface, allows to abstract from actual jo b execution (in the same thread, in a new thread, in a thread pool, etc.)
**ExecutorService** - interface, allows to run Runnable or Callable tasks, manage their lifecycle, and control their execution (shutdown, etc.). Key methods: `.submit()`, `.shutdown()`, `.shutdownNow()`, `.awaitTermination()`, `.invokeAll()`, `.invokeAny()`. A standard way to manage a pool of threads.
**ScheduledExecutorService** - interface that extends ExecutorService, allows to schedule tasks to run after a delay or periodically. Key methods: `.schedule()`, `.scheduleAtFixedRate()`, `.scheduleWithFixedDelay()`.


**Thread** - class that represents a thread of execution in a program. Key methods: `.start()`, `.run()`, `.join()`, `.sleep()`, `.interrupt()`, `.isAlive()`, `.setPriority()`, `.getPriority()`.


**ThreadPoolExecutor** - a class that implements ExecutorService, allows to manage a pool of threads to execute tasks concurrently. Key methods: `.execute()`, `.submit()`, `.shutdown()`, `.shutdownNow()`, `.getPoolSize()`, `.getActiveCount()`, `.getCompletedTaskCount()`.
**ForkJoinPool** - a class that implements ExecutorService, designed for parallelism using the fork/join framework. Key methods: `.invoke()`, `.submit()`, `.shutdown()`, `.shutdownNow()`.
**AutoShutdownDelegatedExecutorService** - a class that wraps an ExecutorService and automatically shuts it down when no longer needed.
**ThreadPerTaskExecutor** - a class that creates a new thread for each task submitted. 

CountDownLatch
CyclicBarrier
Semaphore
Phaser
Exchanger
DelayQueue

## Collections
ConcurrentHashMap
CopyOnWriteArrayList
CopyOnWriteArraySet
ConcurrentLinkedQueue
ConcurrentLinkedDeque
BlockingQueue
LinkedBlockingQueue
ArrayBlockingQueue
ArrayBlockingQueue
DelayQueue

## Atomics
AtomicInteger, AtomicLong, AtomicBoolean, AtomicReference


All the services below implement ExecutorService interface.
<!-- ExecutorService - a service that allows to run from x to y threads (dynamic allocation depending on demand). Key methods: `.submit(task)`, `.shutdown()`,  -->
Executors.newFixedThreadPool - creates a thread pool with a fixed number of threads. Threads are reused for multiple tasks. 
Executors.newWorkStealingPool - creates a thread pool that uses work-stealing algorithm to balance the load among threads. Suitable for parallel tasks.
Executors.newSingleThreadExecutor
Executors.newCachedThreadPool
Executors.newThreadPerTaskExecutor
Executors.newVirtualThreadPerTaskExecutor
Executors.newSingleThreadScheduledExecutor
Executors.newScheduledThreadPool
Executors.unconfigurableExecutorService
Executors.defaultThreadFactory
Executors.privilegedThreadFactory

## Use cases
 - Running asynchronous tasks, managing thread pools, getting result - ExecutorService with Callable and Future
 - Scheduling tasks to run after a delay or periodically - ScheduledExecutorService
 - Waiting for multiple threads to complete before proceeding - CountDownLatch
 - Synchronizing multiple threads at a barrier point - CyclicBarrier (or Phaser for more complex scenarios)
 - Thread-safe collections for concurrent access - ConcurrentHashMap, CopyOnWriteArrayList
 - Producer-consumer scenarios - BlockingQueue implementations like LinkedBlockingQueue or ArrayBlockingQueue
 - Atomic operations on shared variables without locks - AtomicInteger, AtomicLong, etc.
