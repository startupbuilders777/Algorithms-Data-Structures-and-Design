My approach for solving system design in an interview:

Functional Requirements / Use Cases - Spend max of 5 mins here. (Note: not asking questions is red flag)
Non-Functional Requirements/NFR (scalability, latency, CAP, durability etc)- 2 minutes
Back of the envelope (capacity) estimations - Spend max of 2-5 mins.
Generic High Level Design (HLD)-- covering in-flow and outflow of data -- 5-10 mins
Data Model / APIs (as applicable) - 5 mins
Scale each component in HLD depending on NFR - 10 mins
Verify your design / Discuss trade-offs of your design with interviewer (important) - 5 mins
Make your own notes on below items which you can refer to the day before each interview as a quick reference.


###########################################
From https://github.com/jwasham/coding-interview-university#data-structures

You can expect system design questions if you have 4+ years of experience.

Scalability and System Design are very large topics with many topics and resources, since there is a lot to consider when designing a software/hardware system that can scale. Expect to spend quite a bit of time on this.
Considerations:
scalability
Distill large data sets to single values
Transform one data set to another
Handling obscenely large amounts of data
system design
features sets
interfaces
class hierarchies
designing a system under certain constraints
simplicity and robustness
tradeoffs
performance analysis and optimization
 START HERE: The System Design Primer
 System Design from HiredInTech
 How Do I Prepare To Answer Design Questions In A Technical Inverview?
 8 Things You Need to Know Before a System Design Interview
 Algorithm design
 Database Normalization - 1NF, 2NF, 3NF and 4NF (video)
 System Design Interview - There are a lot of resources in this one. Look through the articles and examples. I put some of them below.
 How to ace a systems design interview
 Numbers Everyone Should Know
 How long does it take to make a context switch?
 Transactions Across Datacenters (video)
 A plain English introduction to CAP Theorem
 Consensus Algorithms:
 Paxos - Paxos Agreement - Computerphile (video)
 Raft - An Introduction to the Raft Distributed Consensus Algorithm (video)
 Easy-to-read paper
 Infographic
 Consistent Hashing
 NoSQL Patterns
 Scalability:
You don't need all of these. Just pick a few that interest you.
 Great overview (video)
 Short series:
Clones
Database
Cache
Asynchronism
 Scalable Web Architecture and Distributed Systems
 Fallacies of Distributed Computing Explained
 Pragmatic Programming Techniques
extra: Google Pregel Graph Processing
 Jeff Dean - Building Software Systems At Google and Lessons Learned (video)
 Introduction to Architecting Systems for Scale
 Scaling mobile games to a global audience using App Engine and Cloud Datastore (video)
 How Google Does Planet-Scale Engineering for Planet-Scale Infra (video)
 The Importance of Algorithms
 Sharding
 Scale at Facebook (2012), "Building for a Billion Users" (video)
 Engineering for the Long Game - Astrid Atkinson Keynote(video)
 7 Years Of YouTube Scalability Lessons In 30 Minutes
video
 How PayPal Scaled To Billions Of Transactions Daily Using Just 8VMs
 How to Remove Duplicates in Large Datasets
 A look inside Etsy's scale and engineering culture with Jon Cowie (video)
 What Led Amazon to its Own Microservices Architecture
 To Compress Or Not To Compress, That Was Uber's Question
 Asyncio Tarantool Queue, Get In The Queue
 When Should Approximate Query Processing Be Used?
 Google's Transition From Single Datacenter, To Failover, To A Native Multihomed Architecture
 Spanner
 Machine Learning Driven Programming: A New Programming For A New World
 The Image Optimization Technology That Serves Millions Of Requests Per Day
 A Patreon Architecture Short
 Tinder: How Does One Of The Largest Recommendation Engines Decide Who You'll See Next?
 Design Of A Modern Cache
 Live Video Streaming At Facebook Scale
 A Beginner's Guide To Scaling To 11 Million+ Users On Amazon's AWS
 How Does The Use Of Docker Effect Latency?
 A 360 Degree View Of The Entire Netflix Stack
 Latency Is Everywhere And It Costs You Sales - How To Crush It
 Serverless (very long, just need the gist)
 What Powers Instagram: Hundreds of Instances, Dozens of Technologies
 Cinchcast Architecture - Producing 1,500 Hours Of Audio Every Day
 Justin.Tv's Live Video Broadcasting Architecture
 Playfish's Social Gaming Architecture - 50 Million Monthly Users And Growing
 TripAdvisor Architecture - 40M Visitors, 200M Dynamic Page Views, 30TB Data
 PlentyOfFish Architecture
 Salesforce Architecture - How They Handle 1.3 Billion Transactions A Day
 ESPN's Architecture At Scale - Operating At 100,000 Duh Nuh Nuhs Per Second
 See "Messaging, Serialization, and Queueing Systems" way below for info on some of the technologies that can glue services together
 Twitter:
O'Reilly MySQL CE 2011: Jeremy Cole, "Big and Small Data at @Twitter" (video)
Timelines at Scale
For even more, see "Mining Massive Datasets" video series in the Video Series section.
 Practicing the system design process: Here are some ideas to try working through on paper, each with some documentation on how it was handled in the real world:
review: The System Design Primer
System Design from HiredInTech
cheat sheet
flow:
Understand the problem and scope:
define the use cases, with interviewer's help
suggest additional features
remove items that interviewer deems out of scope
assume high availability is required, add as a use case
Think about constraints:
ask how many requests per month
ask how many requests per second (they may volunteer it or make you do the math)
estimate reads vs. writes percentage
keep 80/20 rule in mind when estimating
how much data written per second
total storage required over 5 years
how much data read per second
Abstract design:
layers (service, data, caching)
infrastructure: load balancing, messaging
rough overview of any key algorithm that drives the service
consider bottlenecks and determine solutions
Exercises:
Design a CDN network: old article
Design a random unique ID generation system
Design an online multiplayer card game
Design a key-value database
Design a picture sharing system
Design a recommendation system
Design a URL-shortener system: copied from above
Design a cache system
