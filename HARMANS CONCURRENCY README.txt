From https://github.com/jwasham/coding-interview-university#data-structures

Parallel Programming
Coursera (Scala)
Efficient Python for High Performance Parallel Computing (video)
Messaging, Serialization, and Queueing Systems
Thrift
Tutorial
Protocol Buffers
Tutorials
gRPC
gRPC 101 for Java Developers (video)
Redis
Tutorial
Amazon SQS (queue)
Amazon SNS (pub-sub)
RabbitMQ
Get Started
Celery
First Steps With Celery
ZeroMQ
Intro - Read The Manual
ActiveMQ
Kafka
MessagePack
Avro


Processes and Threads
 Computer Science 162 - Operating Systems (25 videos):
for processes and threads see videos 1-11
Operating Systems and System Programming (video)
What Is The Difference Between A Process And A Thread?
Covers:
Processes, Threads, Concurrency issues
difference between processes and threads
processes
threads
locks
mutexes
semaphores
monitors
how they work
deadlock
livelock
CPU activity, interrupts, context switching
Modern concurrency constructs with multicore processors
Paging, segmentation and virtual memory (video)
Interrupts (video)
Process resource needs (memory: code, static storage, stack, heap, and also file descriptors, i/o)
Thread resource needs (shares above (minus stack) with other threads in the same process but each has its own pc, stack counter, registers, and stack)
Forking is really copy on write (read-only) until the new process writes to memory, then it does a full copy.
Context switching
How context switching is initiated by the operating system and underlying hardware
 threads in C++ (series - 10 videos)
 concurrency in Python (videos):
 Short series on threads
 Python Threads
 Understanding the Python GIL (2010)
reference
 David Beazley - Python Concurrency From the Ground Up: LIVE! - PyCon 2015
 Keynote David Beazley - Topics of Interest (Python Asyncio)
 Mutex in Python