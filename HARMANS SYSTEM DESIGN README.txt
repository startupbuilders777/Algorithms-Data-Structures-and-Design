1. Feature Expectations [5 min]
	(1) Use Cases
	(2) Scenarios That Will Not Be Covered
	(3) Who Will Use
	(4) How Many Will Use
	(5) Usage Patterns
2. Estimations [5 min]
	(1) Throughput
		  - Queries per second (QPS) for read and write queries.
	(2) Latency
		  - Expected latency for read and write queries.
	(3) Read/Write Ratio
	(4) Traffic Estimates
		  - Write (QPS, volume of data).
		  - Read (QPS, volume of data).
	(5) Storage Estimates
	(6) Memory Estimates
		  - If we are using a cache, what kind of data do we want to store in the cache?
		  - How much RAM and how many machines are needed?
		  - Amount of data to store on disk/SSD.
3. Design Goals [5 min]
	(1) Latency and Throughput Requirements
	(2) Consistency vs. Availability
		  - Weak/strong/eventual consistency.
		  - Failover/replication for availability.
4. High-Level Design [5-8 min]
	(1) APIs for Read/Write Scenarios for Crucial Components
	(2) Database Schema
	(3) Basic Algorithm
	(4) High-Level Design for Read/Write Scenario
5. Deep Dive [10-12 min]
	(1) Scaling the Algorithm
	(2) Scaling Individual Components
		  - Availability, Consistency, and Scale story for each component
		  - Consistency and availability patterns.
	(3) Components to Consider
		  a) DNS
		  b) CDN (Push vs. Pull)
		  c) Load Balancers (Active-Passive, Active-Active, Layer 4, Layer 7)
		  d) Reverse Proxy
		  e) Application Layer Scaling (Microservices, Service Discovery)
		  f) Database options
				- RDBMS: ACID Properties, Primary-Secondary, Primary-Primary, Federation, Sharding, Denormalization, SQL Tuning - Postgres 
					- Use-cases: Structured data with relationships
				- NoSQL: Key-Value, Wide-Column, Document - MongoDB, DynamoDB 
					- Use-cases: Unstructured or semi-structured data
				- Graph: Neo4j, Amazon Neptune 
					- Use-cases: Social networks, knowledge graphs, recommendation systems, and bioinformatics
				- NewSQL: Key-Value with ACID Properties - CockroachDB, Google Spanner, VoltDB 
					- Use-cases: Transaction processing, real-time analytics and IoT device data
				- Time Series: Time-stamped data points - InfluxDB, TimescaleDB, Prometheus
					- Use-cases: IoT sensor data, financial market data, system metrics, and logs
				- Vector: High-dimensional vector data - Pinecone, Weaviate, KDB.AI 
					- Use-cases: Machine learning, similarity search, and recommendation systems
				- Fast lookups:
					  - RAM (Bounded size) => Redis, Memcached.
					  - AP (Unbounded size) => Cassandra, RIAK, Voldemort, DynamoDB (default mode)
					  - CP (Unbounded size) => HBase, MongoDB, Couchbase, DynamoDB (consistent read setting).
		  g) Caches
				- Client caching, CDN caching, Webserver caching, Database caching, Application caching, Cache at query level, Cache at object level.
				- Eviction policies:
					  - Cache aside
					  - Write through
					  - Write behind
					  - Refresh ahead.
		  h) Asynchronism
				- Message queues
				- Task queues
				- Back pressure.
		  i) Communication
				- TCP
				- UDP
				- REST
				- RPC
				- WebSockets
6. Justify [5 min]
	(1) Throughput of Each Layer
	(2) Latency Caused Between Each Layer
	(3) Overall Latency Justification
7. Key Metrics to Measure [3 min]
	(1) Identify key metrics relevant to your system's design
	(2) Define metrics for infrastructure and resources
			- Tools like Graphana with Prometheus, AppDynamics, etc., can be used.
8. System Health Monitoring [2 min]
	(1) Measure app index and latency of microservices
	(2) Tools like New Relic, AppDynamics can be used
			-  Use Grafana with Prometheus or AppDynamics for monitoring
	(3) Canaries - to simulate customer experience and pro-actively detect service degradation
9. Log Systems [2 min]
	(1) Implement tools to gather and visualize metrics	
			- Availability
			- Latency 
			- Throttling 
			- Request Patterns/Volume
	(2) Collect and analyze logs with ELK (Elastic, Logstash, Kibana) or Splunk.

10. Security [2 min]
	(1) Firewall, encryptions at rest and in transit
	(2) TLS
	(3) Authentication, Authorization (AuthN/Z)
	(4) Limited Egress/Ingress
	(5) Principle of least privilege
Extension of: https://leetcode.com/discuss/career/229177/my-system-design-template


##########
Frugal Streaming
Geohash / S2 Geometry
Leaky bucket / Token bucket
Loosy Counting
Operational transformation
Quadtree / Rtree
Ray casting
Reverse index
Rsync algorithm
Trie algorithm
Add Bloom Filters and Count-Min Sketch into the list.



#######

SCRAPE NOTES FROM THIS -> https://github.com/naniroot/naniz-hugo-site/blob/master/content/posts/system_design_interviews_cheatsheet.md

OK L6 INTERVIEW PREP:

Giving back - how I cleared L6 System Design - Part 1
Red Hat  flung
  608 Comments
Part 2 is out : https://www.teamblind.com/post/rBrt5bV8

Edit 5: I know I promised to reply to every comment/question but I left it at 90 comments and now it's 450 and grows faster than I can reply. I'll have to be selective. Sorry.

Edit 4: I will be posting the links of part 2 of SD and future coding post here.

Edit 3: "This is booky knowledge and not real life experience stuff". DDIA is condensed experience. Nishtala et al are as close to *relevant* real life experience as it gets. In fact, I had the feeling some of the experience of my interviewers was irrelevant experience in scaling glorified CRUD apps which had left them stale on knowledge. It's a balance, but I stand by my hints here if your goal is to clear the SD interview. This method got me 3 SH and 1 H in F/G interviews so like it or not, it gets the job done.

Edit 2: Some people feel this is too much detail and you never get asked these things. Wrong. It's all up to you to steer the conversation. The question is pretty vague. The key is to identify unassisted the main pain point(s) of the problem and once you do, dive as deep as possible after you cover the high level design. It is not uncommon that the detail you waved away turns your high-level design to dead end. Everything that I mentioned as examples here were actually brought up in my interviews. Less preparation could get you there but, as I mentioned, I go for the jugular.

Edit: Throw your questions in the comments. I'll get to everyone of them eventually, I promise. DMs also are welcome but it needs to be something you can't put on the comments.

1. Intro

I spent 6 months preparing overall. In the evenings before going to sleep I'd spend some time on Blind to get my hopes up and see a joke or two. There are some assholes here and there on Blind, but the majority are good people. I love you all. So now that I'm 'across the river' I decided to give back and provide some hints specifically on the System Design rounds.

I interviewed at Amazon, Twitter, Google, Facebook, Databricks and Elastic for L6 (or L6 equivalent). I cleared them all with exception of Twitter. I cleared Google and Facebook first then had 'emergency' interviews with the rest of the companies to ensure I was well-placed for the inevitable lowballing dance. One Friday I had half-onsite with Elastic in the morning and half-onsite with Twitter in the evening. I completely messed up the System Design for Twitter and that killed my prospects. I regret that though, as I liked Twitter.

I am not a genius, I'm just a hardworking tenacious guy. Don't let anyone tell you you're not smart enough for L6. It's all about mustering the motivation, channeling the work and executing it well.

I made the decision to try 6 months ago, and I prepared throughout that time while being at work. So my preparation hints will suppose long-term prep. If you're expecting a two-week crash course this is not for you.

I'm conscious some of this intro will read like a flex - it is genuinely not - but I'm sure the context will be helpful to many so I decided to make that tradeoff ;)

2. Content

Where do you start? Get the Designing Data Intensive Applications book. This will fill your theoretical gaps. Read it slowly. *Do not make fake progress*. If you don't understand something stop, use the references, research the subjects, get out of the book and back. There is *nothing* useless for you in that book. Nothing too much. It's all for you cover to cover. Properly grasping that book is half of the whole work. Part 1 is super useful to teach you how to pick data stores. Part 2 will dispel your fears of sharding and choosing a replication mechanism. Part 3 will give you a full idea on how to piece a big system together from smaller systems. The separation of System of record and derived data is key to understand there.

When you think you're done with DDIA, if I wake you up in the middle of the night you should be able to explain to me how LSMT-based databases use Red Black Trees to keep a sorted log in memory and why they do that, and then get back to sleep. I'm serious.

Next up, pay for Grokking the system design interview.

I know that Grokking can be too shallow. Nevertheless, I want you to be able to recite the solutions in Grokking down to the smallest details. Details are more important than the big picture for L6. Do you think your done with typeahead suggestions? Did you mention exponential moving average on how you weigh results? No? Failed.

You can leave the pastebin and bitly and that easy crap uncovered, but I'll need you to recite the other grokking solutions to the smallest details. Take extra care on the geospatial problems. If I show up at your lunch and ask you about quad trees I want you to be able to estimate the index size on the spot. You can't? Not ready yet.

Next up: videos. InfoQ and @scale videos about real life systems. Look up Nathan Bronson and suck up all his videos on Tao, Caches etc. Look up Nishtala and the memcache video. Look up Kulkarni and the Facebook live video. Look up Susheel Aroskar and his Zuul push video <- ultra useful.

Again, never make fake progress. Take this last one: Aroskar's Zuul push. He mentions Apache Netty. Read up on it, understand it. Go deeper, understand epoll and its red black trees. Go deeper and understand the TIMED-WAIT trick in TCP protocol which saves some web sockets connections. Another thing: he mentions some load balancers can't handle websockets. Why? Go figure it out. I did and I impressed my Facebook interviewer as I went deep until I was stopped as the interviewer lost it. L6 means depth, depth, depth. If you draw me a block that says 'Server' and leave it at that I'll slap you back to L3.

Now I went the extra mile. But I wasn't targeting barely making it - I was targeting mind-blowing performance. If you want the hardcore stuff, let me know and I'll give you content for that as well. If you're up for anecdotes, a Google interviewer challenged me on Paxos when I mentioned Spanner to him. I drew the multi-Paxos detailed flow for him with estimated latencies supposing a 5-replica configuration with the pivot replica on north-east US. He smiled and said 'ok' (Strong hire).

Next up, the Google SRE book. Did you say you're not interviewing for SRE? Don't care. You don't need it all. You do need the chapter on Non-Abstract Large System Design. You need to be able to recite that in your sleep. Especially the estimations part. You don't have NALSD interview? *rolls eyes* Don't care, learn the NALSD chapter or fail.

3. Speaking of Estimations

Ultra important. If you can't handle these, you're most likely ducked. How do you prepare for them? Practice power of 10 calculations. See how the NALSD chapter does it. Practice with fake numbers. Drag the units with you and reduce them when dividing/multiplying.

When calculating storage, consider cold storage. Also consider e.g. cassandra needs ±30% free space to do compacting (see, I told you to properly learn LSTMs in Part 1 of DDIA), also keep in mind that only ±85% of the disk space is usable (OS files, formatting tables etc.), also keep in mind that 3-5% of disks die in a year so account for that, also keep in mind that you need to multiply by replication factor, also keep in mind that cassandra says no more than 2TB disks otherwise it gets slow.

Have a strong grasp on the -bound concepts. Is something storage-bound (Log dumps)? Is something cpu-bound (live stream encoding)? Is something RAM bound (in-memory timeline serving)? Once you grasp those you design tiers with separate scaling techniques. That's why you need numbers, not to show off your math but to figure the -bound part and then decide how to scale a specific tier.

I'll wrap up this part here. In the second part I'll get into some sample walk-throughs of some popular questions. Then I'll give a you a detailed plan on how to spend the 45 minutes. I did 9 System Design interviews in the space of 5 weeks so it's fresh in my head.

Now you might think this is a lot of work. It is. It depends on how much you want the job. Want it badly? Throw in the effort then, that's all it takes. Was it worth the effort for me? Yes.

Old TC: 175K USD, New TC: 593k USD


Giving back - how I cleared L6 System Design - Part 2
Red Hat  flung
  48 Comments
This is part 2 of a series. Here is part 1 which has a long intro: https://www.teamblind.com/post/Giving-back---how-I-cleared-L6-System-Design---Part-1-4yufM3RY

0. Preface

Let me touch on some points from Part 1 flood of comments:

Is Nathan Bronson a porn star? I'm not sure if Nathan Bronson of Facebook has a side gig but I meant the other non-porn star Bronson.

YOE? 10. Don't let YOE hold you back though. I've always been somehow the youngest of my peers at something. Age? Early 30s.

Did I have coding rounds? Of course. I will write about that but it will have to wait. Plenty of coding hints on Blind on how to leetcode but I think the shortage is on SDI content so I want to plug that first.

How did I manage the time? I had a method which I'll share eventually.

---- end preface ----

- Back of the envelope calculations (BOTEC)

You expected something else? Sit down... I thought hard and I think this is something many people secretly fear and feel weak on. So since my goal in this effort is to genuinely help people, I think this is where my contribution is most valuable.

You might think why do we even need BOTEC? Short version is to size tiers. What? Size? Tiers? Sizing as a verb here means to estimate how many machines/disks/etc. you will need. Tiers in this context means the typical tier in any system development. For example, a logging/counting system can have three tiers: 1. the collection tier, 2. the aggregation tier and 3. the storage tier.

Another way you can use BOTEC is to figure out if something can fit in one machine or not, mostly memory wise. Most of the time it is self-evident, but 10 seconds of BOTEC should confirm it.

Example: How do you serve the timelines of 1 Million people efficiently? Well how many posts do you expect to have readily available per person? Let's say 10. Ok. If we store the IDs only (say 64 bit) we can use a redis native list to store list of posts:

8 bytes/post * 10 post/timeline=80 bytes/timeline (post goes away)
1 Million timelines * 80 bytes/timeline = 10^6*10^1*8=8*10^(6+1) bytes (timeline goes away)

so 80 million bytes means 80 MB - easily goes into one machine's RAM.

This was an easy example however note two things: always drag the units with you: bytes/post, post/timeline and reduce when possible. It's extremely helpful to not get lost. Second, always use powers of 10 to not miss a zero here and a zero there.

Here's the real life version of that.
-64 bit ID per post - 8 bytes/post
-800 posts/timeline
-500 million timelines
-replication factor of 3

800 posts / timeline * 8 bytes/post = (post goes away) 8*10^2 * 8 bytes/timeline= 64 * 10^2 bytes/timeline

500 million timelines* 64 * 10^2 bytes/timeline = (timeline goes away) 5 * 10^2 * 10^6 * 64 * 10 ^2 bytes= 5*64 * 10 ^(2+6+2) bytes= 300 * 10^10 = 3 *10^12 bytes = 3 TBs

3TBs * 3 (unitless replication factor) = ±10 TB. Considering 64 GB ram/machine out of which 50 can be considered usable you have 10 TB/(50GB/machine) = 10* 10^12 Bytes /(5*10^9)Bytes/machine= 2 *10^3 =2000 Machines (Bytes goes away, 1 over 1 over machine becomes machine).

The temptation here is to stop being pedantic with units but I suggest you don't. These can get messy so stick to it.

This tier can be considered to be memory-bound.

Look up Raffi Krikorian's Timelines at Scale on infoQ to see him talk about this at more length.

Usually you have different tiers with different scaling mechanisms. Do your calculations early to get a feeling of what is going to be the bounding factor, then plan ahead to scale that accordingly.

Here's an example of a QPS-bound tier. You're told you'll have to log 100 billion events per day. Ok. That sounds like a lot. Let's turn that into per second. Is that a calculator you took out? *takes it and throws out the window*

100 * 10^9 events/day [divided by] 86400 seconds/day
round to a convenient number
100 * 10^9 events/day [divided by] 80000 seconds/day=
10^11/(8*10^4) events/second (1 over day goes away)=1/8 *(10^7)Events/sec= 10/8 Million QPS=~1.2 Million QPS

This looks to be a QPS bound tier. To size it, divide it by some (numerically convenient) number that can be handled by one machine and you get the number of machines.

Same goes for storage sizing. Keep in mind the replication factor in storage and the amount of time you will be storing for. Look up datastax capacity planning for cassandra numbers, they're super useful.

And last but not least here are some numbers I used to reference quite often. They're taken from various sources and I used them without causing dropped jaws, so it should be safe.

- Compress 1KB with Zippy - 0.003 ms
- Send 1KB over 1Gbps 0.01 ms
- Read 4MB of sequential memory is 1 ms
- Read 4MB of sequential SSD is 20 ms
- Read 4MB of sequential HDD is 80 ms
- One single disk seek is 10 ms
- Inter-datacenter roundtrip 150ms
- Inter-datacenter bandwidth ±100 Gbps
- Video is roughly 10 MB/minute
- A server can have 100-150GB of usable RAM
- Cassandra is best used with 1-2TB storage / node
- A 1Gbps link can handle max of 125 MB /s
- Cassandra cluster can handle 10-15000 reqs/s/node and grows linearly with number of nodes
- 720p video - 1.5Mbps
- 340 video - 150Kbps

End of part 2


Thanks a lot for putting both the posts together. Would you mind sharing a structured (non-negotiable/not to be skipped list) for your approach? The reason am asking for this because reading about SDI seems to be an endless exercise. For example, from the limited knowledge I have the way I see it there are multiple aspects to it:

* Distributed systems basics (most of which might be covered in DDIA) - anything else you'd suggest?
* Advanced concepts for distributed systems (e.g. how exactly consensus work under the covers) - what exactly should we look at? Any relevant blog links?
* Algorithms which are pivotal to specific systems (Quadtree/Geohash for Uber, Google Maps, Tinder etc; Tries for Typeahead suggestions; Leaky bucket for rate limiting etc) - Was there any list which you went through which you can share?
* White papers which explain famous technologies under the hood (e.g. GFS, BigTable, Kafka, Paxos, Dynamo etc) - Can you share your list ?
* If we break down a simple system into multiple parts, each aspect has a lot of technologies that can come in : Client-Server (Netty/Poly..) ; Load Balancing (NGINX /Netscaler..), Data Processing (Spark, Flink..); Storage (Cassandra, DynamoDB etc. ) - Again did you have a list that you went through or you can share?

I feel overwhelmed with the amount of breadth this topic has and feel lost at times. Any guidance would help!




#############################################
WRITE NOTES FOR THIS -> https://hakibenita.com/sql-tricks-application-dba

My approach for solving system design in an interview:

Functional Requirements / Use Cases - Spend max of 5 mins here. 
            (Note: not asking questions is red flag)

Non-Functional Requirements/NFR (scalability, latency, CAP, durability etc)- 2 minutes

Back of the envelope (capacity) estimations - Spend max of 2-5 mins.

Generic High Level Design (HLD)-- covering in-flow and outflow of data -- 5-10 mins

Data Model / APIs (as applicable) - 5 mins

Scale each component in HLD depending on NFR - 10 mins

Verify your design / Discuss trade-offs of your design with interviewer (important) - 5 mins

Make your own notes on below items which you can refer to 
    the day before each interview as a quick reference.



ALSO READ: https://github.com/donnemartin/system-design-primer#system-design-topics-start-here
and: https://tianpan.co/hacking-the-software-engineer-interview/

Virual onsite 5: System Design - Usual system design for Google scale image storage. 


But expect to go deep with numbers (throughput, latency, storage...). 


The interviewer went very deep with how to decide the number of servers, 
database throughput capacity, cache memory and so on. A lot of discussion on tradeoffs.
######################################3
REST IDEMPOTENCY:

Idempotent REST APIs
In the context of REST APIs, when making multiple identical requests has the same effect as making a single request – then that REST API is called idempotent.

When you design REST APIs, you must realize that API consumers can make mistakes. Users can write client code in such a way that there can be duplicate requests coming to the API.

These duplicate requests may be unintentional as well as intentional some time (e.g. due to timeout or network issues). You have to design fault-tolerant APIs in such a way that duplicate requests do not leave the system unstable.

An idempotent HTTP method is an HTTP method that can be called many times without different outcomes. It would not matter if the method is called only once, or ten times over. The result should be the same.
Idempotence essentially means that the result of a successfully performed request is independent of the number of times it is executed. For example, in arithmetic, adding zero to a number is an idempotent operation.

Idempotency with HTTP Methods
If you follow REST principles in designing API, you will have automatically idempotent REST APIs for GET, PUT, DELETE, HEAD, OPTIONS and TRACE HTTP methods. Only POST APIs will not be idempotent.

POST is NOT idempotent.
GET, PUT, DELETE, HEAD, OPTIONS and TRACE are idempotent.
Let’s analyze how the above HTTP methods end up being idempotent – and why POST is not.

HTTP POST
Generally – not necessarily – POST APIs are used to create a new resource on server. So when you invoke the same POST request N times, you will have N new resources on the server. So, POST is not idempotent.

HTTP GET, HEAD, OPTIONS and TRACE
GET, HEAD, OPTIONS and TRACE methods NEVER change the resource state on server. They are purely for retrieving the resource representation or meta data at that point of time. So invoking multiple requests will not have any write operation on server, so GET, HEAD, OPTIONS and TRACE are idempotent.

HTTP PUT
Generally – not necessarily – PUT APIs are used to update the resource state. If you invoke a PUT API N times, the very first request will update the resource; then rest N-1 requests will just overwrite the same resource state again and again – effectively not changing anything. Hence, PUT is idempotent.

HTTP DELETE
When you invoke N similar DELETE requests, first request will delete the resource and response will be 200 (OK) or 204 (No Content). Other N-1 requests will return 404 (Not Found). Clearly, the response is different from first request, but there is no change of state for any resource on server side because original resource is already deleted. So, DELETE is idempotent.

Please keep in mind if some systems may have DELETE APIs like this:

DELETE /item/last
In the above case, calling operation N times will delete N resources – hence DELETE is not idempotent in this case. In this case, a good suggestion might be to change the above API to POST – because POST is not idempotent.

POST /item/last
Now, this is closer to HTTP spec – hence more REST compliant.

References:

Rfc 2616
SO Thread
#############################################################################

- For system design: System Design Primer and Designing Data Intensive Applications.

- If you have time, read Grokking the system design interview but don't let it fool you. You will NOT have 
time during an interview to go through all the steps listed on Grokking. Instead, ask the interviewers what 
areas of the design they'd like to focus on, they will almost always tell you. Following the grokking blueprint 
might actually work against you and make you seem unready for a senior role.



- Get very good at designing APIs. They came up in all of my system design interviews. You don't need to focus 
on a particular implementation (HTTP, RPC, etc). Just be ready to discuss inputs, outputs, error handling, etc.


####################################
HORIZONTAL AND VERTICAL SCALING:

Horizontal and Vertical Scaling

Discussing about the concept of horizontal and vertical Scaling in System Design along with their 
characteristics, gains and drwaback

System Design Concepts

May 16, 2020
Saurav Prateek | Author
  Need for ScalingWhen you have a small system having less amount of load then it’s quite easy to maintain. You don’t have to think much about increasing computing power, handling large numbers of read/write requests, your server getting crashed and much more. But what happens when the load over your system increases. Now you will need to worry about the above mentioned problems. While designing a system one should keep in mind the amount of load which it has to bear and should meet the end user’s requirements at the same time. This is probably called Scaling. When such a situation occurs you have two things under your sleeves. Either you can increase the computation power of your underlying system or else you can increase the number of systems so that it can handle the increasing load. This brings us to our topic i.e. horizontal and vertical scaling.Horizontal ScalingIn order to handle the increasing load you can increase the number of machines. This process is known as Horizontal Scaling. The process has its own advantages and drawbacks. When you increase the number of machines you will need to have a Load Balancer which will ensure that the requests coming to your system are distributed uniformly over all the machines. We will be discussing Load Balancers in the next article. But your system can scale really well even the amount of load increases. You will just be needed to put extra machines according to the increasing load and route the requests efficiently to all the systems.

If you are scaling horizontally then your system will have following characteristics :

Resilient : You system will be resilient. It will be able to handle large amounts of loads or requests and will be able to recover in less time in case of any failure.

Network Calls : As you are using multiple machines to handle your load, you will be needing a procedure to set up a contact between these machines. You will be required to set up a network or a remote procedure call among the machines.

Need of Load Balancer : As discussed earlier you will be needing a load balancer to distribute the requests among the machines. As you are using multiple machines, there can be scenarios where one machine gets a huge number of requests and another may be sitting idle. In order to reduce this uneven distribution of requests we will have to use a Load Balancer which will ensure that every machine gets an equal amount of requests and your system can handle huge loads efficiently.

Scales Well : Your system will have the capacity to handle an increasing number of loads. You can increase the number of machines according to the load and route the requests evenly to these machines.

Vertical ScalingTo handle the increasing load you could increase the computation capacity of your existing system. This involves adding more processing power, more storage, more memory etc. to your system. In this part you won’t be needing any load balancer or network calls as there is a single system with a huge computation capacity handling the entire load. This can be fast. But there can be a scenario where you won’t be able to add more resources to your underlying machine because of increasing cost or anything and buying identical machines will be a cheaper and better idea.

If you are scaling vertically your system will be having following categories :

Consistent : As your system has a single machine the data will be consistent as it will be stored at one place. You can avoid Data Inconsistency when using Vertical Scaling.

Single Point of Failure : Being dependent on a single machine can cause a single point of failure. If there is any fault in the machine and it is unable to handle the requests in the anticipated way, then your entire system can fail.

Inter Process Communication : With vertical scaling you can achieve inter process communication. The processes can communicate with each other and synchronize their actions. This can make the system handle requests faster.

So, which is better: Horizontal Scaling or Vertical. I guess it can vary in different situations. Suppose your system is needed to handle the amount of load which can be handled by adding some extra computation power to the existing machine then you can go with Vertical Scaling. You will be able to achieve a consistent system and moreover won’t be needing any Load Balancers or Network Calls. But if adding computation power to the machine seems unfeasible then you can add more identical machines instead. Then you will be able to handle an increasing amount of loads with ease having a resilient system.

Another idea can be to build a Hybrid System taking the advantage of both the techniques. Each machine has high computation power and has multiple such machines to handle the increasing loads.
##############################3

##################################
DISTRIBUTED PROGRAMMING:
    (Read https://github.com/henryr/cap-faq)
    
    Performance vs scalability
    A service is scalable if it results in increased performance in a manner proportional to resources added. 
    Generally, increasing performance means serving more units of work, but it can also be to handle 
    larger units of work, such as when datasets grow.1

    Another way to look at performance vs scalability:

    If you have a performance problem, your system is slow for a single user.
    If you have a scalability problem, your system is fast for a single user but slow under heavy load.
    Source(s) and further reading
    A word on scalability
    Scalability, availability, stability, patterns
    Latency vs throughput
    Latency is the time to perform some action or to produce some result.

    Throughput is the number of such actions or results per unit of time.

    Generally, you should aim for maximal throughput with acceptable latency.

    Source(s) and further reading
    Understanding latency vs throughput
    Availability vs consistency
    CAP theorem

    Source: CAP theorem revisited

    In a distributed computer system, you can only support two of the following guarantees:

    Consistency - Every read receives the most recent write or an error
    Availability - Every request receives a response, without guarantee that it contains the most recent version of the information
    Partition Tolerance - The system continues to operate despite arbitrary partitioning due to network failures
    Networks aren't reliable, so you'll need to support partition tolerance.
        You'll need to make a software tradeoff between consistency and availability.

    CP - consistency and partition tolerance
    Waiting for a response from the partitioned node might result in a timeout error. CP is a good choice if your 
    business needs require atomic reads and writes.

    AP - availability and partition tolerance
    Responses return the most readily available version of the data available on any node, which might not be the latest. 
    Writes might take some time to propagate when the partition is resolved.

    AP is a good choice if the business needs allow for eventual consistency or when the system needs to continue working despite external errors.


    Consistency patterns
    With multiple copies of the same data, we are faced with options on how to synchronize them so 
    clients have a consistent view of the data. Recall the definition of consistency from the CAP theorem - 
    Every read receives the most recent write or an error.

    Weak consistency
    After a write, reads may or may not see it. A best effort approach is taken.

    This approach is seen in systems such as memcached. Weak consistency works well in real time use cases such as VoIP, video chat, and realtime multiplayer games. For example, if you are on a phone call and lose reception for a few seconds, when you regain connection you do not hear what was spoken during connection loss.

    Eventual consistency
    After a write, reads will eventually see it (typically within milliseconds). Data is replicated asynchronously.

    This approach is seen in systems such as DNS and email. Eventual consistency works well in highly available systems.

    Strong consistency
    After a write, reads will see it. Data is replicated synchronously.

    This approach is seen in file systems and RDBMSes. Strong consistency works well in systems that need transactions.

    Source(s) and further reading
    Transactions across data centers (https://snarfed.org/transactions_across_datacenters_io.html)


    Availability patterns
    There are two complementary patterns to support high availability: fail-over and replication.

    Fail-over
    Active-passive
    With active-passive fail-over, heartbeats are sent between the active and the passive 
    server on standby. If the heartbeat is interrupted, the passive server takes over the active's IP address and resumes service.

    The length of downtime is determined by whether the passive server is already running in 'hot' standby or whether it needs to start up from 'cold' standby. Only the active server handles traffic.

    Active-passive failover can also be referred to as master-slave failover.

    Active-active
    In active-active, both servers are managing traffic, spreading the load between them.

    If the servers are public-facing, the DNS would need to know about the public IPs of both servers. If the servers are internal-facing, application logic would need to know about both servers.

    Active-active failover can also be referred to as master-master failover.

    Disadvantage(s): failover
    Fail-over adds more hardware and additional complexity.
    There is a potential for loss of data if the active system fails before any newly written data can be replicated to the passive.
    Replication
    Master-slave and master-master
    This topic is further discussed in the Database section:

    Master-slave replication
    Master-master replication
    Availability in numbers
    Availability is often quantified by uptime (or downtime) as a percentage of time the service is available. Availability is generally measured in number of 9s--a service with 99.99% availability is described as having four 9s.

    99.9% availability - three 9s
    Duration	Acceptable downtime
    Downtime per year	8h 45min 57s
    Downtime per month	43m 49.7s
    Downtime per week	10m 4.8s
    Downtime per day	1m 26.4s
    99.99% availability - four 9s
    Duration	Acceptable downtime
    Downtime per year	52min 35.7s
    Downtime per month	4m 23s
    Downtime per week	1m 5s
    Downtime per day	8.6s
    Availability in parallel vs in sequence
    If a service consists of multiple components prone to failure, the service's overall availability depends on whether the components are in sequence or in parallel.

    In sequence
    Overall availability decreases when two components with availability < 100% are in sequence:

    Availability (Total) = Availability (Foo) * Availability (Bar)
    If both Foo and Bar each had 99.9% availability, their total availability in sequence would be 99.8%.

    In parallel
    Overall availability increases when two components with availability < 100% are in parallel:

    Availability (Total) = 1 - (1 - Availability (Foo)) * (1 - Availability (Bar))
    If both Foo and Bar each had 99.9% availability, their total availability in parallel would be 99.9999%.
########################################
DNS AND CDN:

Domain name system

Source: DNS security presentation

A Domain Name System (DNS) translates a domain name such as www.example.com to an IP address.

DNS is hierarchical, with a few authoritative servers at the top level. 
Your router or ISP provides information about which DNS server(s) to contact 
when doing a lookup. Lower level DNS servers cache mappings, which could become 
stale due to DNS propagation delays. DNS results can also be cached by your browser 
or OS for a certain period of time, determined by the time to live (TTL).

NS record (name server) - Specifies the DNS servers for your domain/subdomain.
MX record (mail exchange) - Specifies the mail servers for accepting messages.
A record (address) - Points a name to an IP address.
CNAME (canonical) - Points a name to another name or CNAME (example.com to www.example.com) or to an A record.
Services such as CloudFlare and Route 53 provide managed DNS services. Some DNS services can route traffic through various methods:

Weighted round robin
Prevent traffic from going to servers under maintenance
Balance between varying cluster sizes
A/B testing
Latency-based
Geolocation-based
Disadvantage(s): DNS
Accessing a DNS server introduces a slight delay, although mitigated by caching described above.
DNS server management could be complex and is generally managed by governments, ISPs, and large companies.
DNS services have recently come under DDoS attack, preventing users from accessing websites such as Twitter without knowing Twitter's IP address(es).
Source(s) and further reading
DNS architecture
Wikipedia
DNS articles

Source: Why use a CDN

A content delivery network (CDN) is a globally distributed network of proxy servers, serving content from locations closer to the user. Generally, static files such as HTML/CSS/JS, photos, and videos are served from CDN, although some CDNs such as Amazon's CloudFront support dynamic content. The site's DNS resolution will tell clients which server to contact.

Serving content from CDNs can significantly improve performance in two ways:

Users receive content from data centers close to them
Your servers do not have to serve requests that the CDN fulfills
Push CDNs
Push CDNs receive new content whenever changes occur on your server. You take full responsibility for providing content, uploading directly to the CDN and rewriting URLs to point to the CDN. You can configure when content expires and when it is updated. Content is uploaded only when it is new or changed, minimizing traffic, but maximizing storage.

Sites with a small amount of traffic or sites with content that isn't often updated work well with push CDNs. Content is placed on the CDNs once, instead of being re-pulled at regular intervals.

Pull CDNs
Pull CDNs grab new content from your server when the first user requests the content. You leave the content on your server and rewrite URLs to point to the CDN. This results in a slower request until the content is cached on the CDN.

A time-to-live (TTL) determines how long content is cached. Pull CDNs minimize 
storage space on the CDN, but can create redundant traffic if files expire and are pulled before they have actually changed.

Sites with heavy traffic work well with pull CDNs, as traffic is spread out more evenly with only recently-requested content remaining on the CDN.

Disadvantage(s): CDN
CDN costs could be significant depending on traffic, although this should be weighed with additional costs you would incur not using a CDN.
Content might be stale if it is updated before the TTL expires it.
CDNs require changing URLs for static content to point to the CDN.
Source(s) and further reading
Globally distributed content delivery
The differences between push and pull CDNs
Wikipedia

################################################3

LOAD BALANCING

    Load BalancersExploring the concept of Load Balancers in System Design. Discussing their use cases, drawbacks and going through different types of Load Balancers available.System Design Concepts
    May 16, 2020
    Saurav Prateek | Author
    IntroductionWhen you have multiple Servers handling the requests coming from the end users then it is essential for the requests to be heavily distributed among the servers. Here the Load Balancer comes in play. A Load Balancer acts as a layer between the server and the end user which distributes the requests evenly to all the servers. With a load balancer present no user request can directly go to any servers, it first needs to go through a load balancer and then it can be further be directed to one of the servers.Designing an approach to route requests from the Load BalancerLet’s design a basic mechanism to route the requests to one of the servers when it reaches the Load Balancer.

    Suppose we have N Servers with us and a Load Balancer L which routes the requests to these N servers.

    Now a request with a Request ID R1 reaches the Load Balancer. Our Load Balancer L could possibly use some hashed function in order to hash the request id R1 and then further use this hashed request id say h(R1) to further route it to the server. Now we need to do some maths in order to bring this hashed number in the range of our server numbers N so that it can be routed further. We can do it possible by taking modulo N of the hashed id and then use that number as a server id.

    Let’s See how :

    R1 : Request ID coming to get served
    h : Evenly distributed Hash Function
    h(R1) : Hashed Request ID
    N : Number of Servers


    Suppose we have 8 Servers having IDs : S0, S1, S2 ……, S7

    A request with ID 17 reaches the Load Balancer.

    Now the hash function is used to hash that ID and we get 110 as the hashed value h(17) = 110

    Now we can bring it in the range of our server numbers 110 % 8 = 6

    So we can route this request to Server S2.

    In this way we can distribute all the requests coming to our Load Balancer evenly to all the servers. But is it an optimal approach? Yes it distributes the requests evenly but what if we need to increase the number of our servers. Increasing the server will change the destination servers of all the incoming requests. What if we were storing the cache related to that request in its destination server? Now as that request is no longer routed to the earlier server, our entire cache can go in trash probably. Think!

    There are various Load Balancers which are used and can be essential to know about.L4 Load BalancerL4 Load Balancer routes the request on the basis of the address information of the incoming requests. It does not inspect the content of the request. The Layer 4 (L4) Load Balancer makes the routing decisions based on address information extracted from the first few packets in the TCP stream.L7 Load BalancerL7 Load Balancer routes the request on the basis of the packet content. There can be dedicated servers which could serve the requests based on their content, like URLs, Images, Graphics and Video contents. You can set up the system in such a way that the static contents are served by one dedicated server and requests demanding certain information which needs a db call can be served by another dedicated server.



    Load balancers distribute incoming client requests to computing resources such as application servers and databases. In each case, the load balancer returns the response from the computing resource to the appropriate client. Load balancers are effective at:

    Preventing requests from going to unhealthy servers
    Preventing overloading resources
    Helping to eliminate a single point of failure
    Load balancers can be implemented with hardware (expensive) or with software such as HAProxy.

    Additional benefits include:

    SSL termination - Decrypt incoming requests and encrypt server responses so backend servers do not have to perform these potentially expensive operations
    Removes the need to install X.509 certificates on each server
    Session persistence - Issue cookies and route a specific client's requests to same instance if the web apps do not keep track of sessions
    To protect against failures, it's common to set up multiple load balancers, either in active-passive or active-active mode.

    Load balancers can route traffic based on various metrics, including:

    Random
    Least loaded
    Session/cookies
    Round robin or weighted round robin
    Layer 4
    Layer 7
    Layer 4 load balancing
    Layer 4 load balancers look at info at the transport layer to decide how to distribute requests. Generally, this involves the source, destination IP addresses, and ports in the header, but not the contents of the packet. Layer 4 load balancers forward network packets to and from the upstream server, performing Network Address Translation (NAT).

    Layer 7 load balancing
    Layer 7 load balancers look at the application layer to decide how to distribute requests. This can involve contents of the header, message, and cookies. Layer 7 load balancers terminate network traffic, reads the message, makes a load-balancing decision, then opens a connection to the selected server. For example, a layer 7 load balancer can direct video traffic to servers that host videos while directing more sensitive user billing traffic to security-hardened servers.

    At the cost of flexibility, layer 4 load balancing requires less time and computing resources than Layer 7, although the performance impact can be minimal on modern commodity hardware.

    Horizontal scaling
    Load balancers can also help with horizontal scaling, improving performance and availability. Scaling out using commodity machines is more cost efficient and results in higher availability than scaling up a single server on more expensive hardware, called Vertical Scaling. It is also easier to hire for talent working on commodity hardware than it is for specialized enterprise systems.

    Disadvantage(s): horizontal scaling
    Scaling horizontally introduces complexity and involves cloning servers
    Servers should be stateless: they should not contain any user-related data like sessions or profile pictures
    Sessions can be stored in a centralized data store such as a database (SQL, NoSQL) or a persistent cache (Redis, Memcached)
    Downstream servers such as caches and databases need to handle more simultaneous connections as upstream servers scale out
    Disadvantage(s): load balancer
    The load balancer can become a performance bottleneck if it does not have enough resources or if it is not configured properly.
    Introducing a load balancer to help eliminate a single point of failure results in increased complexity.
    A single load balancer is a single point of failure, configuring multiple load balancers further increases complexity.
    Source(s) and further reading
    NGINX architecture
    HAProxy architecture guide
    Scalability
    Wikipedia
    Layer 4 load balancing
    Layer 7 load balancing
    ELB listener config


########################################

REVERSE PROXIES

    A reverse proxy is a web server that centralizes internal services and provides unified interfaces to the public. 
    Requests from clients are forwarded to a server that can fulfill it before the reverse proxy 
    returns the server's response to the client.

    Additional benefits include:

    Increased security - Hide information about backend servers, blacklist IPs, limit number of connections per client
    Increased scalability and flexibility - Clients only see the reverse proxy's IP, allowing you to scale servers or change their configuration
    SSL termination - Decrypt incoming requests and encrypt server responses so backend servers do not 
        have to perform these potentially expensive operations
    Removes the need to install X.509 certificates on each server
    Compression - Compress server responses
    Caching - Return the response for cached requests
    Static content - Serve static content directly
    HTML/CSS/JS
    Photos
    Videos
    Etc
    Load balancer vs reverse proxy
    Deploying a load balancer is useful when you have multiple servers. Often, load balancers route traffic 
    to a set of servers serving the same function.

    Reverse proxies can be useful even with just one web server or application server, 
    opening up the benefits described in the previous section.

    Solutions such as NGINX and HAProxy can support both layer 7 reverse proxying and load balancing.
    Disadvantage(s): reverse proxy
    Introducing a reverse proxy results in increased complexity.
    A single reverse proxy is a single point of failure, configuring multiple 
    reverse proxies (ie a failover) further increases complexity.
    Source(s) and further reading
    Reverse proxy vs load balancer
    NGINX architecture
    HAProxy architecture guide
    Wikipedia





######################################
ASYNCHRONISM:

    Asynchronism

    Source: Intro to architecting systems for scale

    Asynchronous workflows help reduce request times for expensive operations that 
    would otherwise be performed in-line. They can also help by doing time-consuming 
    work in advance, such as periodic aggregation of data.

    Message queues
    Message queues receive, hold, and deliver messages. If an operation is too slow to 
    perform inline, you can use a message queue with the following workflow:

    An application publishes a job to the queue, then notifies the user of job status
    A worker picks up the job from the queue, processes it, then signals the job is complete
    The user is not blocked and the job is processed in the background. During this time, the 
    client might optionally do a small amount of processing to make it seem like the task has 
    completed. For example, if posting a tweet, the tweet could be instantly posted to your 
    timeline, but it could take some time before your tweet is actually delivered to all of your followers.

    Redis is useful as a simple message broker but messages can be lost.

    RabbitMQ is popular but requires you to adapt to the 'AMQP' protocol and manage your own nodes.

    Amazon SQS is hosted but can have high latency and has the possibility of messages being delivered twice.

    Task queues
    Tasks queues receive tasks and their related data, runs them, then delivers their results. 
    They can support scheduling and can be used to run computationally-intensive jobs in the background.

    Celery has support for scheduling and primarily has python support.

    Back pressure
    If queues start to grow significantly, the queue size can become larger than memory, 
    resulting in cache misses, disk reads, and even slower performance. Back pressure can 
    help by limiting the queue size, thereby maintaining a high throughput rate and good 
    response times for jobs already in the queue. Once the queue fills up, clients get a 
    server busy or HTTP 503 status code to try again later. Clients can retry the request 
    at a later time, perhaps with exponential backoff.

    Disadvantage(s): asynchronism
    Use cases such as inexpensive calculations and realtime workflows might be 
    better suited for synchronous operations, as introducing queues can add delays and complexity.
    Source(s) and further reading
    It's all a numbers game
    Applying back pressure when overloaded
    Little's law
    What is the difference between a message queue and a task queue?




###################################################################
NETWORKS AND COMMUNICATION:

TCP:

    TCP is a connection-oriented protocol over an IP network. Connection is established and 
    terminated using a handshake. All packets sent are guaranteed to reach the destination 
    in the original order and without corruption through:

    Sequence numbers and checksum fields for each packet
    Acknowledgement packets and automatic retransmission
    If the sender does not receive a correct response, it will resend the packets. If there 
    are multiple timeouts, the connection is dropped. TCP also implements flow control and 
    congestion control. These guarantees cause delays and generally result in less efficient transmission than UDP.

    To ensure high throughput, web servers can keep a large number of TCP connections open, 
    resulting in high memory usage. It can be expensive to have a large number of open 
    connections between web server threads and say, a memcached server. Connection pooling 
    can help in addition to switching to UDP where applicable.

    TCP is useful for applications that require high reliability but are less time critical. 
    Some examples include web servers, database info, SMTP, FTP, and SSH.

    Use TCP over UDP when:

    You need all of the data to arrive intact
    You want to automatically make a best estimate use of the network throughput
    User datagram protocol (UDP)

    Source: How to make a multiplayer game

    UDP is connectionless. Datagrams (analogous to packets) are guaranteed only at the datagram level. 
    Datagrams might reach their destination out of order or not at all. UDP does not support 
    congestion control. Without the guarantees that TCP support, UDP is generally more efficient.

    UDP can broadcast, sending datagrams to all devices on the subnet. This is useful with DHCP 
    because the client has not yet received an IP address, thus preventing a way for 
    TCP to stream without the IP address.

    UDP is less reliable but works well in real time use cases such as VoIP, 
    video chat, streaming, and realtime multiplayer games.

    Use UDP over TCP when:

    You need the lowest latency
    Late data is worse than loss of data
    You want to implement your own error correction

########################################################################

CACHE STRATEGIES:

    Cache

    Source: Scalable system design patterns

    Caching improves page load times and can reduce the load on your servers and databases. In this model, the dispatcher will first lookup if the request has been made before and try to find the previous result to return, in order to save the actual execution.

    Databases often benefit from a uniform distribution of reads and writes across its partitions. Popular items can skew the distribution, causing bottlenecks. Putting a cache in front of a database can help absorb uneven loads and spikes in traffic.

    Client caching
    Caches can be located on the client side (OS or browser), server side, or in a distinct cache layer.

    CDN caching
    CDNs are considered a type of cache.

    Web server caching
    Reverse proxies and caches such as Varnish can serve static and dynamic content directly. Web servers can also cache requests, returning responses without having to contact application servers.

    Database caching
    Your database usually includes some level of caching in a default configuration, optimized for a generic use case. Tweaking these settings for specific usage patterns can further boost performance.

    Application caching
    In-memory caches such as Memcached and Redis are key-value stores between your application and your data storage. Since the data is held in RAM, it is much faster than typical databases where data is stored on disk. RAM is more limited than disk, so cache invalidation algorithms such as least recently used (LRU) can help invalidate 'cold' entries and keep 'hot' data in RAM.

    Redis has the following additional features:

    Persistence option
    Built-in data structures such as sorted sets and lists
    There are multiple levels you can cache that fall into two general categories: database queries and objects:

    Row level
    Query-level
    Fully-formed serializable objects
    Fully-rendered HTML
    Generally, you should try to avoid file-based caching, as it makes cloning and auto-scaling more difficult.

    Caching at the database query level
    Whenever you query the database, hash the query as a key and store the result to the cache. This approach suffers from expiration issues:

    Hard to delete a cached result with complex queries
    If one piece of data changes such as a table cell, you need to delete all cached queries that might include the changed cell
    Caching at the object level
    See your data as an object, similar to what you do with your application code. Have your application assemble the dataset from the database into a class instance or a data structure(s):

    Remove the object from cache if its underlying data has changed
    Allows for asynchronous processing: workers assemble objects by consuming the latest cached object
    Suggestions of what to cache:

    User sessions
    Fully rendered web pages
    Activity streams
    User graph data
    When to update the cache
    Since you can only store a limited amount of data in cache, you'll need to determine which cache update strategy works best for your use case.

    Cache-aside

    Source: From cache to in-memory data grid

    The application is responsible for reading and writing from storage. The cache does not interact with storage directly. The application does the following:

    Look for entry in cache, resulting in a cache miss
    Load entry from the database
    Add entry to cache
    Return entry
    def get_user(self, user_id):
        user = cache.get("user.{0}", user_id)
        if user is None:
            user = db.query("SELECT * FROM users WHERE user_id = {0}", user_id)
            if user is not None:
                key = "user.{0}".format(user_id)
                cache.set(key, json.dumps(user))
        return user
    Memcached is generally used in this manner.

    Subsequent reads of data added to cache are fast. Cache-aside is also referred to as lazy loading. Only requested data is cached, which avoids filling up the cache with data that isn't requested.

    Disadvantage(s): cache-aside
    Each cache miss results in three trips, which can cause a noticeable delay.
    Data can become stale if it is updated in the database. This issue is mitigated by setting a time-to-live (TTL) which forces an update of the cache entry, or by using write-through.
    When a node fails, it is replaced by a new, empty node, increasing latency.
    Write-through

    Source: Scalability, availability, stability, patterns

    The application uses the cache as the main data store, reading and writing data to it, while the cache is responsible for reading and writing to the database:

    Application adds/updates entry in cache
    Cache synchronously writes entry to data store
    Return
    Application code:

    set_user(12345, {"foo":"bar"})
    Cache code:

    def set_user(user_id, values):
        user = db.query("UPDATE Users WHERE id = {0}", user_id, values)
        cache.set(user_id, user)
    Write-through is a slow overall operation due to the write operation, but subsequent reads of just written data are fast. Users are generally more tolerant of latency when updating data than reading data. Data in the cache is not stale.

    Disadvantage(s): write through
    When a new node is created due to failure or scaling, the new node will not cache entries until the entry is updated in the database. Cache-aside in conjunction with write through can mitigate this issue.
    Most data written might never be read, which can be minimized with a TTL.
    Write-behind (write-back)

    Source: Scalability, availability, stability, patterns

    In write-behind, the application does the following:

    Add/update entry in cache
    Asynchronously write entry to the data store, improving write performance
    Disadvantage(s): write-behind
    There could be data loss if the cache goes down prior to its contents hitting the data store.
    It is more complex to implement write-behind than it is to implement cache-aside or write-through.
    Refresh-ahead

    Source: From cache to in-memory data grid

    You can configure the cache to automatically refresh any recently accessed cache entry prior to its expiration.

    Refresh-ahead can result in reduced latency vs read-through if the cache can accurately predict which items are likely to be needed in the future.

    Disadvantage(s): refresh-ahead
    Not accurately predicting which items are likely to be needed in the future can result in reduced performance than without refresh-ahead.
    Disadvantage(s): cache
    Need to maintain consistency between caches and the source of truth such as the database through cache invalidation.
    Cache invalidation is a difficult problem, there is additional complexity associated with when to update the cache.
    Need to make application changes such as adding Redis or memcached.
    Source(s) and further reading
    From cache to in-memory data grid
    Scalable system design patterns
    Introduction to architecting systems for scale
    Scalability, availability, stability, patterns
    Scalability
    AWS ElastiCache strategies
    Wikipedia


    Write Back, Write Through and Write Around CacheWhen updates are made in your system then you will need to ensure that the data present in your cache is also updated accordingly, in order to avoid any data inconsistency in future. Your cache should hold the latest version of your data.

    Write Through Cache : In this method the data is updated in the cache as well as in the database eventually at the same time. The process is considered to be completed when the data has been successfully written in cache as well as in the backend database. The method ensures the data consistency but you could face latency as the data is needed to be written at two places.

    Write Around Cache : In this method data is updated in the backend database only without writing it in the cache. The process is not bothered by a successful write into the cache. As the data is written to the database the process is considered to be completed. You could face a cache miss for the data recently written into the database.

    Write Back Cache : In this method data is updated in the cache only. The completion of the process is not backed by a successful write to the backend database. The data may be written in the backend but that’s not the part of this process and hence not essential here. If the cache fails then your system may experience data loss.
    Distributed Cache - MemcachedA distributed cache functions like a simple cache but in order to provide a large storage space they are distributed over multiple machines. When a request hits your system then your system first checks the distributed cache for the presence of required data items. The cache maps every request to a machine id where that can be further processed. It then fetches the response from that particular machine.

    Memcached is an example of a distributed cache. It maintains the data in a key-value pair format. You could hit the cache with a particular key and it will return the corresponding value to that key. It may return false or null value if that key is not present in the cache. Memcache also follows LRU (Least Recently Used) Policy for keeping the data items in the cache.


###################################################################

REST VS RPC:
    ( https://github.com/donnemartin/system-design-primer#latency-numbers-every-programmer-should-know)
    Remote procedure call (RPC)

    Source: Crack the system design interview

    In an RPC, a client causes a procedure to execute on a different address space, usually a remote server. 
    The procedure is coded as if it were a local procedure call, abstracting away the details of how to 
    communicate with the server from the client program. Remote calls are usually slower and less 
    reliable than local calls so it is helpful to distinguish RPC calls from local calls. 
    Popular RPC frameworks include Protobuf, Thrift, and Avro.

    RPC is a request-response protocol:

    Client program - Calls the client stub procedure. The parameters are pushed onto the stack like a local procedure call.
    Client stub procedure - Marshals (packs) procedure id and arguments into a request message.
    Client communication module - OS sends the message from the client to the server.
    Server communication module - OS passes the incoming packets to the server stub procedure.
    Server stub procedure - Unmarshalls the results, calls the server procedure matching the procedure id and passes the given arguments.
    The server response repeats the steps above in reverse order.
    Sample RPC calls:

    GET /someoperation?data=anId

    POST /anotheroperation
    {
    "data":"anId";
    "anotherdata": "another value"
    }

    RPC is focused on exposing behaviors. RPCs are often used for performance reasons with 
    internal communications, as you can hand-craft native calls to better fit your use cases.

    Choose a native library (aka SDK) when:

    You know your target platform.
    You want to control how your "logic" is accessed.
    You want to control how error control happens off your library.
    Performance and end user experience is your primary concern.
    HTTP APIs following REST tend to be used more often for public APIs.

    Disadvantage(s): RPC
    RPC clients become tightly coupled to the service implementation.
    A new API must be defined for every new operation or use case.
    It can be difficult to debug RPC.
    You might not be able to leverage existing technologies out of the box. For example, it might 
    require additional effort to ensure RPC calls are properly cached on caching servers such as Squid.
    Representational state transfer (REST)
    REST is an architectural style enforcing a client/server model where the client acts on a set of 
    resources managed by the server. The server provides a representation of resources and actions 
    that can either manipulate or get a new representation of resources. All communication must 
    be stateless and cacheable.

    There are four qualities of a RESTful interface:

    Identify resources (URI in HTTP) - use the same URI regardless of any operation.
    Change with representations (Verbs in HTTP) - use verbs, headers, and body.
    Self-descriptive error message (status response in HTTP) - Use status codes, don't reinvent the wheel.
    HATEOAS (HTML interface for HTTP) - your web service should be fully accessible in a browser.
    Sample REST calls:

    GET /someresources/anId

    PUT /someresources/anId
    {"anotherdata": "another value"}
    REST is focused on exposing data. It minimizes the coupling between client/server and is often used for public HTTP APIs. REST uses a more generic and uniform method of exposing resources through URIs, representation through headers, and actions through verbs such as GET, POST, PUT, DELETE, and PATCH. Being stateless, REST is great for horizontal scaling and partitioning.

    Disadvantage(s): REST
    With REST being focused on exposing data, it might not be a good fit if resources are not naturally organized or 
    accessed in a simple hierarchy. For example, returning all updated records from the past hour matching a particular 
    set of events is not easily expressed as a path. With REST, it is likely to be implemented with a combination of URI 
    path, query parameters, and possibly the request body.
    REST typically relies on a few verbs (GET, POST, PUT, DELETE, and PATCH) which sometimes doesn't fit your use case. 
    For example, moving expired documents to the archive folder might not cleanly fit within these verbs.
    Fetching complicated resources with nested hierarchies requires multiple round trips between the client and server to 
    render single views, e.g. fetching content of a blog entry and the comments on that entry. For mobile applications 
    operating in variable network conditions, these multiple roundtrips are highly undesirable.
    Over time, more fields might be added to an API response and older clients will receive all new data fields, 
    even those that they do not need, as a result, it bloats the payload size and leads to larger latencies.

    RPC and REST calls comparison
    Operation	RPC	REST
    Signup	
    POST /signup	
    POST /persons
    
    
    Resign	POST /resign
    {
    "personid": "1234"
    }	DELETE /persons/1234
    Read a person	GET /readPerson?personid=1234	GET /persons/1234
    Read a person’s items list	GET /readUsersItemsList?personid=1234	GET /persons/1234/items
    Add an item to a person’s items	POST /addItemToUsersItemsList
    {
    "personid": "1234";
    "itemid": "456"
    }	POST /persons/1234/items
    {
    "itemid": "456"
    }
    Update an item	POST /modifyItem
    {
    "itemid": "456";
    "key": "value"
    }	PUT /items/456
    {
    "key": "value"
    }
    Delete an item	POST /removeItem
    {
    "itemid": "456"
    }	DELETE /items/456

    Source: Do you really know why you prefer REST over RPC

##############################################
CONSISTENT HASHING PART 1: 


Consistent Hashing

Exploring the concept of Consistent Hashing in System Design. Discussing the 
requirement for consistent hashing and drawbacks related to this conceptSystem Design Concepts
May 16, 2020
Saurav Prateek | Author
  IntroductionConsistent Hashing is a concept extensively used in Load Balancing. In a typical Load Balancer we use hashing in order to map the requests to their corresponding Servers. Here when we had to add or remove any Server then the Hash Values of all the earlier requests got modified which caused our cache to become obsolete. In order to avoid this problem we use the concept of Consistent Hashing. Using consistent hashing in our system we can avoid the change in the hash function of all the earlier Requests whenever any new server is added or removed. It allows only a small amount of requests to change hence making our cache very much in use.Understanding the ProblemSuppose you have a Load Balancer which distributes your requests among three servers : S1, S2 and S3. We have used a hash function to direct the requests to one of the servers in the system. The entire method used looks like this :

Server ID (Destination) = h( Request ID ) % 3

Here we can clearly see that the request id of the incoming request is hashed by a hash function and then we did a modulo of the result with the total number of servers which is 3 in this case.

Suppose 5 Requests come initially then they will be directed in the following order :

[ Request ID ] 3 ---> h(3) = 101 ---> 101%3 = 2 [ Server ID ]
[ Request ID ] 12 ---> h(12) = 39 ---> 39%3 = 0 [ Server ID ]
[ Request ID ] 41 ---> h(41) = 22 ---> 22%3 = 1 [ Server ID ]
[ Request ID ] 62 ---> h(62) = 98 ---> 98%3 = 2 [ Server ID ]
[ Request ID ] 92 ---> h(92) = 10 ---> 10%3 = 1 [ Server ID ]

As you can see, by using the earlier method we directed our requests to 3 Servers. Now what happens when an extra server is added to the System. Now we have 4 Servers with us and this will have a large impact on the address of the destination servers of the earlier requests.

When the same 5 requests reach the System, it is directed in the following manner.

[ Request ID ] 3 ---> h(3) = 101 ---> 101%4 = 1 [ Server ID ]
[ Request ID ] 12 ---> h(12) = 39 ---> 39%4 = 3 [ Server ID ]
[ Request ID ] 41 ---> h(41) = 22 ---> 22%4 = 2 [ Server ID ]
[ Request ID ] 62 ---> h(62) = 98 ---> 98%4 = 2 [ Server ID ]
[ Request ID ] 92 ---> h(92) = 10 ---> 10%4 = 2 [ Server ID ]

We can clearly observe that the destinations of almost all the requests have been changed. 
Hence by adding or removing even one server will cause the requests to be directed 
to a completely different server. This can have huge drawbacks. Suppose if a server 
stored some critical data in cache about a certain request in the hope of re-using 
it when that request visits again. Now the entire data in the cache is obsolete.

So our current method of Hashing is not quite reliable in order to direct the requests to our multiple servers. We need something which could cause a very less amount of change in the destination of the requests when a new server is added or removed. Here Consistent Hashing comes into play. Let’s see how it solves our current problem.Discussing Consistent HashingIt is a method which is independent of the number of servers present in the System. It hashes all the Servers to a hash ID which is plotted on a circular ring. This ring like structure allows a very minimal change in the requests when a server is added or removed.

Suppose our Hash Function returns a value in the range of 0 to N. The ring starts from 0 and ends at N. Now each Server present is hashed using this function and is plotted over the ring. Suppose we have three servers S1, S2 and S3 and have 5 requests coming to the system.

We used our hash function say H1 and hashed every server and request to get the values like this :

Server : S1 ------> H1( S1 ) ------> Hashed Value : HS1
Server : S2 ------> H1( S2 ) ------> Hashed Value : HS2
Server : S3 ------> H1( S3 ) ------> Hashed Value : HS3

Request : R1 ------> H1( R1 ) ------> Hashed Value : HR1
Request : R2 ------> H1( R2 ) ------> Hashed Value : HR2
Request : R3 ------> H1( R3 ) ------> Hashed Value : HR3
Request : R4 ------> H1( R4 ) ------> Hashed Value : HR4
Request : R5 ------> H1( R5 ) ------> Hashed Value : HR5

Now they can be plotted on the circular pie as described below :

Here every Request is served by the Server which is adjacent to it when we move in a clockwise manner. 
Hence Requests R1 and R2 will be served by Server S1, Requests R3 and R4 will be served by the Server S2 
and Request R5 will be served by Server S3.

Suppose Server S1 got shut down due to some issue. Then Requests R1 and R2 which was earlier served by S1 
will now be served by server S2 leaving all the other requests completely unchanged. This caused only 
those requests to change which was earlier served by the obsolete server. Rest of the requests were completely 
unaffected by the change in the number of servers. This is one of the major advantages of using Consistent Hashing 
in a system.

Skewed Load in Consistent HashingEarlier when me removed Server S1 from the system Requests R1 and R2 were 
served by Server S2. Now requests R1, R2, R3 and R4 are served by Server S2. That means 4 out of 5 requests are 
served by a single server. The entire load is skewed on a single server and this can cause that System to wear out or shut down.

What can be done to avoid this problem here. Instead of using a single hash function we can possibly 
use multiple Hash Functions to hash the servers. Suppose we used K hash functions. Now each server will 
have K points on the circular ring and then it will be less likely to have a skewed load over a single server.

We can select the value of K accordingly in order to reduce the chance of having skewed load 
to none. Possibly log(N) or log(M) where N is the Number of available Servers and M is the number of Requests.

############################################

CONSISTENT HASHING PART 2:

Here’s a problem. I have a set of keys and values. I also have some servers for a key-value store. This could be memcached, Redis, MySQL, whatever. I want to distribute the keys across the servers so I can find them again. And I want to do this without having to store a global directory.
One solution is called mod-N hashing.
First, choose a hash function to map a key (string) to an integer. Your hash function should be fast. This tends to rule out cryptographic ones like SHA-1 or MD5. Yes they are well distributed but they are also too expensive to compute — there are much cheaper options available. Something like MurmurHash is good, but there are slightly better ones out there now. Non-cryptographic hash functions like xxHash, MetroHash or SipHash1–3 are all good replacements.
If you have N servers, you hash your key with the hash function and take the resulting integer modulo N.
 server := serverList[hash(key) % N]
This setup has a number of advantages. First, it’s very easy to explain. It’s also very cheap to compute. The modulo can be expensive but it’s almost certainly cheaper than hashing the key. If your N is a power of two then you can just mask off the lower bits. (This is a great way to shard a set of locks or other in-memory data structure.)
What are the downsides of this approach? The first is that if you change the number of servers, almost every key will map somewhere else. This is bad.
Let’s consider what an “optimal” function would do here.
When adding or removing servers, only 1/nth of the keys should move.
Don’t move any keys that don’t need to move.
To expand on the first point, if we’re moving from 9 servers to 10, then the new server should be filled with 1/10th of all the keys. And those keys should be evenly chosen from the 9 “old” servers. And keys should only move to the new server, never between two old servers. Similarly, if we need to remove a server (say, because it crashed), then the keys should be evenly distributed across the remaining live servers.
Luckily, there’s a paper that solves this. In 1997, the paper “Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web” was released. This paper described the approach used by Akamai in their distributed content delivery network.
It took until 2007 for the ideas to seep into the popular consciousness. That year saw two works published:
last.fm’s Ketama memcached client.
Dynamo: Amazon’s Highly Available Key-value Store
These cemented consistent hashing’s place as a standard scaling technique. It’s now used by Cassandra, Riak, and basically every other distributed system that needs to distribute load over servers.
This algorithm is the popular ring-based consistent hashing. You may have seen a “points-on-the-circle” diagram. When you do an image search for “consistent hashing”, this is what you get:
Image for post
It scrolls on like this for a while
You can think of the circle as all integers 0 ..2³²-1. The basic idea is that each server is mapped to a point on a circle with a hash function. To lookup the server for a given key, you hash the key and find that point on the circle. Then you scan forward until you find the first hash value for any server.
In practice, each server appears multiple times on the circle. These extra points are called “virtual nodes”, or “vnodes”. This reduces the load variance among servers. With a small number of vnodes, different servers could be assigned wildly different numbers of keys.
(A brief note on terminology. The original consistent hashing paper called servers “nodes”. Papers will generally talk about“nodes”, “servers”, or “shards”. This article will use all three interchangeably.)
One of the other nice things about ring hashing is that the algorithm is straight-forward. Here’s a simple implementation taken from groupcache (slightly modified for clarity):
To add the list of nodes to the ring hash, each one is hashed m.replicas times with slightly different names ( 0 node1, 1 node1, 2 node1, …). The hash values are added to the m.nodes slice and the mapping from hash value back to node is stored in m.hashMap. Finally the m.nodes slice is sorted so we can use a binary search during lookup.
func (m *Map) Add(nodes ...string) {
    for _, n := range nodes {
        for i := 0; i < m.replicas; i++ {
            hash := int(m.hash([]byte(strconv.Itoa(i) + " " + n)))
            m.nodes = append(m.nodes, hash)
            m.hashMap[hash] = n
        }
    }
    sort.Ints(m.nodes)
}
To see which node a given key is stored on, it’s hashed into an integer. The sorted nodes slice is searched to see find the smallest node hash value larger than the key hash (with a special case if we need to wrap around to the start of the circle). That node hash is then looked up in the map to determine the node it came from.
func (m *Map) Get(key string) string {
    hash := int(m.hash([]byte(key)))
    idx := sort.Search(len(m.keys),
        func(i int) bool { return m.keys[i] >= hash }
    )
    if idx == len(m.keys) {
        idx = 0
    }
    return m.hashMap[m.keys[idx]]
}
Ketama is a memcached client that uses a ring hash to shard keys across server instances. I needed a compatible Go implementation and came across this problem.
What’s the Go equivalent of this line of C?
unsigned int k_limit = floorf(pct * 40.0 * ketama->numbuckets);
It’s a trick question: you can’t answer it in isolation. You need to know these types and also C’s promotion rules:
float floorf(float x);
unsigned int numbuckets;
float pct;
The answer is this:
limit := int(float32(float64(pct) * 40.0 * float64(numbuckets)))
And the reason is because of C’s arithmetic promotion rules and because the 40.0 constant is a float64.
And once I had this sorted out for my go-ketama implementation, I immediately wrote my own ring hash library (libchash) which didn’t depend on floating point round-off error for correctness. My library is also slightly faster because it doesn’t use MD5 for hashing.
Lesson: avoid implicit floating point conversions, and probably floating point in general, if you’re building anything that needs to be cross-language.
End of interlude.
“Are we done?” OR “Why Is This Still a Research Topic?”
Ring hashing presents a solution to our initial problem. Case closed? Not quite. Ring hashing still has some problems.
First, the load distribution across the nodes can still be uneven. With 100 replicas (“vnodes”) per server, the standard deviation of load is about 10%. The 99% confidence interval for bucket sizes is 0.76 to 1.28 of the average load (i.e., total keys / number of servers). This sort of variability makes capacity planning tricky. Increasing the number of replicas to 1000 points per server reduces the standard deviation to ~3.2%, and a much smaller 99% confidence interval of 0.92 to 1.09.
This comes with significant memory cost. For 1000 nodes, this is 4MB of data, with O(log n) searches (for n=1e6) all of which are processor cache misses even with nothing else competing for the cache.







##############################################

Latency numbers every programmer should know

    Latency Comparison Numbers
    --------------------------
    L1 cache reference                           0.5 ns
    Branch mispredict                            5   ns
    L2 cache reference                           7   ns                      14x L1 cache
    Mutex lock/unlock                           25   ns
    Main memory reference                      100   ns                      20x L2 cache, 200x L1 cache
    Compress 1K bytes with Zippy            10,000   ns       10 us
    Send 1 KB bytes over 1 Gbps network     10,000   ns       10 us
    Read 4 KB randomly from SSD*           150,000   ns      150 us          ~1GB/sec SSD
    Read 1 MB sequentially from memory     250,000   ns      250 us
    Round trip within same datacenter      500,000   ns      500 us
    Read 1 MB sequentially from SSD*     1,000,000   ns    1,000 us    1 ms  ~1GB/sec SSD, 4X memory
    HDD seek                            10,000,000   ns   10,000 us   10 ms  20x datacenter roundtrip
    Read 1 MB sequentially from 1 Gbps  10,000,000   ns   10,000 us   10 ms  40x memory, 10X SSD
    Read 1 MB sequentially from HDD     30,000,000   ns   30,000 us   30 ms 120x memory, 30X SSD
    Send packet CA->Netherlands->CA    150,000,000   ns  150,000 us  150 ms

    Notes
    -----
    1 ns = 10^-9 seconds
    1 us = 10^-6 seconds = 1,000 ns
    1 ms = 10^-3 seconds = 1,000 us = 1,000,000 ns

    Handy metrics based on numbers above:

    Read sequentially from HDD at 30 MB/s
    Read sequentially from 1 Gbps Ethernet at 100 MB/s
    Read sequentially from SSD at 1 GB/s
    Read sequentially from main memory at 4 GB/s
    6-7 world-wide round trips per second
    2,000 round trips per second within a data center


    Lets multiply all these durations by a billion:

    Magnitudes:

    Minute:
    L1 cache reference                  0.5 s         One heart beat (0.5 s)
    Branch mispredict                   5 s           Yawn
    L2 cache reference                  7 s           Long yawn
    Mutex lock/unlock                   25 s          Making a coffee
    Hour:
    Main memory reference               100 s         Brushing your teeth
    Compress 1K bytes with Zippy        50 min        One episode of a TV show (including ad breaks)
    Day:
    Send 2K bytes over 1 Gbps network   5.5 hr        From lunch to end of work day
    Week
    SSD random read                     1.7 days      A normal weekend
    Read 1 MB sequentially from memory  2.9 days      A long weekend
    Round trip within same datacenter   5.8 days      A medium vacation
    Read 1 MB sequentially from SSD    11.6 days      Waiting for almost 2 weeks for a delivery
    Year
    Disk seek                           16.5 weeks    A semester in university
    Read 1 MB sequentially from disk    7.8 months    Almost producing a new human being
    The above 2 together                1 year
    Decade
    Send packet CA->Netherlands->CA     4.8 years     Average time it takes to complete a bachelor's degree
###################################################
SQL VS NoSQL

Reasons for SQL:

Structured data
Strict schema
Relational data
Need for complex joins
Transactions
Clear patterns for scaling
More established: developers, community, code, tools, etc
Lookups by index are very fast
Reasons for NoSQL:

Semi-structured data
Dynamic or flexible schema
Non-relational data
No need for complex joins
Store many TB (or PB) of data
Very data intensive workload
Very high throughput for IOPS
Sample data well-suited for NoSQL:

Rapid ingest of clickstream and log data
Leaderboard or scoring data
Temporary data, such as a shopping cart
Frequently accessed ('hot') tables
Metadata/lookup tables

###############################################

NOSQL:

NoSQL
NoSQL is a collection of data items represented in a key-value store, document store, wide column store, or a graph database. Data is denormalized, and joins are generally done in the application code. Most NoSQL stores lack true ACID transactions and favor eventual consistency.

BASE is often used to describe the properties of NoSQL databases. In comparison with the CAP Theorem, BASE chooses availability over consistency.

Basically available - the system guarantees availability.
Soft state - the state of the system may change over time, even without input.
Eventual consistency - the system will become consistent over a period of time, given that the system doesn't receive input during that period.
In addition to choosing between SQL or NoSQL, it is helpful to understand which type of NoSQL database best fits your use case(s). We'll review key-value stores, document stores, wide column stores, and graph databases in the next section.

Key-value store
Abstraction: hash table

A key-value store generally allows for O(1) reads and writes and is often backed by memory or SSD. Data stores can maintain keys in lexicographic order, allowing efficient retrieval of key ranges. Key-value stores can allow for storing of metadata with a value.

Key-value stores provide high performance and are often used for simple data models or for rapidly-changing data, such as an in-memory cache layer. Since they offer only a limited set of operations, complexity is shifted to the application layer if additional operations are needed.

A key-value store is the basis for more complex systems such as a document store, and in some cases, a graph database.

Source(s) and further reading: key-value store
Key-value database
Disadvantages of key-value stores
Redis architecture
Memcached architecture
Document store
Abstraction: key-value store with documents stored as values

A document store is centered around documents (XML, JSON, binary, etc), where a document stores all information for a given object. Document stores provide APIs or a query language to query based on the internal structure of the document itself. Note, many key-value stores include features for working with a value's metadata, blurring the lines between these two storage types.

Based on the underlying implementation, documents are organized by collections, tags, metadata, or directories. Although documents can be organized or grouped together, documents may have fields that are completely different from each other.

Some document stores like MongoDB and CouchDB also provide a SQL-like language to perform complex queries. DynamoDB supports both key-values and documents.

Document stores provide high flexibility and are often used for working with occasionally changing data.

Source(s) and further reading: document store
Document-oriented database
MongoDB architecture
CouchDB architecture
Elasticsearch architecture
Wide column store

Source: SQL & NoSQL, a brief history

Abstraction: nested map ColumnFamily<RowKey, Columns<ColKey, Value, Timestamp>>

A wide column store's basic unit of data is a column (name/value pair). A column can be grouped in column families (analogous to a SQL table). Super column families further group column families. You can access each column independently with a row key, and columns with the same row key form a row. Each value contains a timestamp for versioning and for conflict resolution.

Google introduced Bigtable as the first wide column store, which influenced the open-source HBase often-used in the Hadoop ecosystem, and Cassandra from Facebook. Stores such as BigTable, HBase, and Cassandra maintain keys in lexicographic order, allowing efficient retrieval of selective key ranges.

Wide column stores offer high availability and high scalability. They are often used for very large data sets.

Source(s) and further reading: wide column store
SQL & NoSQL, a brief history
Bigtable architecture
HBase architecture
Cassandra architecture
Graph database

Source: Graph database

Abstraction: graph

In a graph database, each node is a record and each arc is a relationship between two nodes. Graph databases are optimized to represent complex relationships with many foreign keys or many-to-many relationships.

Graphs databases offer high performance for data models with complex relationships, such as a social network. They are relatively new and are not yet widely-used; it might be more difficult to find development tools and resources. Many graphs can only be accessed with REST APIs.

Source(s) and further reading: graph
Graph database
Neo4j
FlockDB
Source(s) and further reading: NoSQL
Explanation of base terminology
NoSQL databases a survey and decision guidance
Scalability
Introduction to NoSQL
NoSQL patterns




###################################################

RDBMS:


    Relational database management system (RDBMS)
    A relational database like SQL is a collection of data items organized in tables.

    ACID is a set of properties of relational database transactions.

    Atomicity - Each transaction is all or nothing
    Consistency - Any transaction will bring the database from one valid state to another
    Isolation - Executing transactions concurrently has the same results as if the transactions were executed serially
    Durability - Once a transaction has been committed, it will remain so
    There are many techniques to scale a relational database: master-slave replication, master-master replication, federation, 
    sharding, denormalization, and SQL tuning.

    Master-slave replication
    The master serves reads and writes, replicating writes to one or more slaves, which serve only reads. 
    Slaves can also replicate to additional slaves in a tree-like fashion. If the master goes offline, 
    the system can continue to operate in read-only mode until a slave is promoted to a master or a new master is provisioned.


    Source: Scalability, availability, stability, patterns

    Disadvantage(s): master-slave replication
    Additional logic is needed to promote a slave to a master.
    See Disadvantage(s): replication for points related to both master-slave and master-master.
    Master-master replication
    Both masters serve reads and writes and coordinate with each other on writes. If either master goes down, the system can continue to operate with both reads and writes.


    Source: Scalability, availability, stability, patterns

    Disadvantage(s): master-master replication
    You'll need a load balancer or you'll need to make changes to your application logic to determine where to write.
    Most master-master systems are either loosely consistent (violating ACID) or have increased write latency due to synchronization.
    Conflict resolution comes more into play as more write nodes are added and as latency increases.
    See Disadvantage(s): replication for points related to both master-slave and master-master.
    Disadvantage(s): replication
    There is a potential for loss of data if the master fails before any newly written data can be replicated to other nodes.
    Writes are replayed to the read replicas. If there are a lot of writes, the read replicas can get bogged down with replaying writes and can't do as many reads.
    The more read slaves, the more you have to replicate, which leads to greater replication lag.
    On some systems, writing to the master can spawn multiple threads to write in parallel, whereas read replicas only support writing sequentially with a single thread.
    Replication adds more hardware and additional complexity.
    Source(s) and further reading: replication
    Scalability, availability, stability, patterns
    Multi-master replication
    Federation

    Source: Scaling up to your first 10 million users

    Federation (or functional partitioning) splits up databases by function. For example, instead of a single, monolithic database, you could have three databases: forums, users, and products, resulting in less read and write traffic to each database and therefore less replication lag. Smaller databases result in more data that can fit in memory, which in turn results in more cache hits due to improved cache locality. With no single central master serializing writes you can write in parallel, increasing throughput.

    Disadvantage(s): federation
    Federation is not effective if your schema requires huge functions or tables.
    You'll need to update your application logic to determine which database to read and write.
    Joining data from two databases is more complex with a server link.
    Federation adds more hardware and additional complexity.
    Source(s) and further reading: federation
    Scaling up to your first 10 million users
    Sharding

    Source: Scalability, availability, stability, patterns

    Sharding distributes data across different databases such that each database can only manage a subset of the data. Taking a users database as an example, as the number of users increases, more shards are added to the cluster.

    Similar to the advantages of federation, sharding results in less read and write traffic, less replication, and more cache hits. Index size is also reduced, which generally improves performance with faster queries. If one shard goes down, the other shards are still operational, although you'll want to add some form of replication to avoid data loss. Like federation, there is no single central master serializing writes, allowing you to write in parallel with increased throughput.

    Common ways to shard a table of users is either through the user's last name initial or the user's geographic location.

    Disadvantage(s): sharding
    You'll need to update your application logic to work with shards, which could result in complex SQL queries.
    Data distribution can become lopsided in a shard. For example, a set of power users on a shard could result in increased load to that shard compared to others.
    Rebalancing adds additional complexity. A sharding function based on consistent hashing can reduce the amount of transferred data.
    Joining data from multiple shards is more complex.
    Sharding adds more hardware and additional complexity.
    Source(s) and further reading: sharding
    The coming of the shard
    Shard database architecture
    Consistent hashing
    Denormalization
    Denormalization attempts to improve read performance at the expense of some write performance. Redundant copies of the data are written in multiple tables to avoid expensive joins. Some RDBMS such as PostgreSQL and Oracle support materialized views which handle the work of storing redundant information and keeping redundant copies consistent.

    Once data becomes distributed with techniques such as federation and sharding, managing joins across data centers further increases complexity. Denormalization might circumvent the need for such complex joins.

    In most systems, reads can heavily outnumber writes 100:1 or even 1000:1. A read resulting in a complex database join can be very expensive, spending a significant amount of time on disk operations.

    Disadvantage(s): denormalization
    Data is duplicated.
    Constraints can help redundant copies of information stay in sync, which increases complexity of the database design.
    A denormalized database under heavy write load might perform worse than its normalized counterpart.
    Source(s) and further reading: denormalization
    Denormalization
    SQL tuning
    SQL tuning is a broad topic and many books have been written as reference.

    It's important to benchmark and profile to simulate and uncover bottlenecks.

    Benchmark - Simulate high-load situations with tools such as ab.
    Profile - Enable tools such as the slow query log to help track performance issues.
    Benchmarking and profiling might point you to the following optimizations.

    Tighten up the schema
    MySQL dumps to disk in contiguous blocks for fast access.
    Use CHAR instead of VARCHAR for fixed-length fields.
    CHAR effectively allows for fast, random access, whereas with VARCHAR, you must find the end of a string before moving onto the next one.
    Use TEXT for large blocks of text such as blog posts. TEXT also allows for boolean searches. Using a TEXT field results in storing a pointer on disk that is used to locate the text block.
    Use INT for larger numbers up to 2^32 or 4 billion.
    Use DECIMAL for currency to avoid floating point representation errors.
    Avoid storing large BLOBS, store the location of where to get the object instead.
    VARCHAR(255) is the largest number of characters that can be counted in an 8 bit number, often maximizing the use of a byte in some RDBMS.
    Set the NOT NULL constraint where applicable to improve search performance.
    Use good indices
    Columns that you are querying (SELECT, GROUP BY, ORDER BY, JOIN) could be faster with indices.
    Indices are usually represented as self-balancing B-tree that keeps data sorted and allows searches, sequential access, insertions, and deletions in logarithmic time.
    Placing an index can keep the data in memory, requiring more space.
    Writes could also be slower since the index also needs to be updated.
    When loading large amounts of data, it might be faster to disable indices, load the data, then rebuild the indices.
    Avoid expensive joins
    Denormalize where performance demands it.
    Partition tables
    Break up a table by putting hot spots in a separate table to help keep it in memory.
    Tune the query cache
    In some cases, the query cache could lead to performance issues.
    Source(s) and further reading: SQL tuning
    Tips for optimizing MySQL queries
    Is there a good reason i see VARCHAR(255) used so often?
    How do null values affect performance?
    Slow query log





########################################
READ THIS -> https://systemsthatscale.org/


#####################################################
MONOLITHIC VS MICROSERVICES:

Monolithic ArchitectureA system implementing Monolithic Architecture stores all of its components at a single place or in a single system. Suppose we are designing an E-commerce Application which has multiple components like : Product Catalogue, User Profile, User Cart and many more components.
Being a Monolithic Architecture the system stores all its components at a single machine. Such a system can have following characteristics :

Fast : A system following a monolithic architecture can be comparatively fast as no network call or Remote Procedure Call (RPC) is required between multiple components. As the components are located at a single machine, hence a simple function call can get the job done. And these function calls are much faster than the network calls, hence the entire system can be fast.

Large and Complex Systems : As all the components of the System are stored at a single place or a single machine, hence the system can get large. And if there are large components in a system then the entire system can become complex and hard to understand. This can cause several maintenance issues in future.

Hard to Scale : Systems having a monolithic architecture can be hard or costlier to scale. They can take up unnecessary resources which can be saved by adopting a Microservice architecture. Earlier we had an E-commerce application with multiple components. Suppose there is a sale going on and this has a huge load over the Product Catalogue section. A large number of users are trying to access the Catalogue. So you need to scale that component so that it can respond to the increasing load. You will be increasing the number of servers according to the increasing load. But in this architecture this will be a costly approach as all the components are stored in a single system and increasing the number of servers will indeed scale all the components present in the System.

This is one of the major drawbacks with Monolithic Architecture. The component which is not facing a spike in the load is also scaled along unnecessarily.

Hefty Deployments : Suppose you made a small change in the code base of the User Profile component of the E-commerce application. At the time of deployment you will need to deploy the entire system which can be costly and time taking as well. This is also a major drawback as all the components reside in a single machine.

Hard to Understand : Suppose a new developer joins your team and is assigned to work on one of the components of your system. As your system implements a monolithic architecture, hence all the components are clubbed in a single machine and that new developer has to go through the entire system in order to get a clear understanding of a single component. This makes a monolithic architecture hard and time taking to understand. The system may appear to be complex to a new individual. It can appear to be unstructured.

Complex Testing process : A monolithic architecture is complex to test. Suppose you have performed a small change in one of the components and have deployed the system. Then you will be needed to test the entire system as there is a chance of having a bug introduced anywhere in the system as all the components are clubbed into a single machine.
Microservice ArchitectureA system implementing Microservice Architecture has multiple components known as microservice which are loosely coupled and functioning independently. Suppose we have an E-commerce Application having multiple components as discussed earlier. In a microservice architecture all these components are stored in separate machines or servers which communicate remotely with each other through a Network Call or a Remote Procedure Call (RPC). These components function individually and solely as a separate entity. They come up together to form an entire system.

Basically a large system is broken down into multiple components which are then stored in distinct machines and are set to operate independently from each other. All these components function as an independent and complete system.A Microservice Architecture have following characteristics :

Network Calls required : In a microservice architecture all the components are stored in different machines and hence requires a medium to communicate. These components communicate with each other through a set of APIs or Network Calls. An individual component receives the request from the network call , processes it and returns the response again through a network call.

Highly Scalable : A microservice architecture is highly scalable. As all the components function individually hence it is easier to scale a particular component according to the requirements. In an E-commerce application when there is a sudden spike in the load on the Catalogue component during the time of sale then we can easily scale that particular component only. As these components are stored in separate machines, we can increase the count of the machines which hold that particular component according to the increasing load. The rest of the components which didn’t receive any load hike are kept untouched. Hence using the resources efficiently.

Easy to Test : A system implementing a microservice architecture is easy to test. Suppose we need to make a change in the User Profile component of the E-commerce application. In that case we can make the change in that system and test it independently for various test cases and once we are satisfied with the testing we can deploy the changes. We are not required to test the entire system in case of a microservice architecture. We only need to test that particular component or microservice in which the change has been made.

Easy to Understand : A microservice architecture is easy to understand. As all the components function independently and are not dependent on others in any ways. Hence when a new developer comes up and has to work on a particular component then in that case he/she is not required to go through the entire system. They can simply understand the working of that particular component or microservice which they have been assigned to work on. As these components are loosely coupled, allow the developers to work independently on the services.
####################################################

APPLICATION LAYER AND MICROSERVICES:

Application layer

Source: Intro to architecting systems for scale

Separating out the web layer from the application layer (also known as platform layer) 
allows you to scale and configure both layers independently. Adding a new API results in 
adding application servers without necessarily adding additional web servers. 
The single responsibility principle advocates for small and autonomous services 
that work together. Small teams with small services can plan more aggressively for rapid growth.

Workers in the application layer also help enable asynchronism.

Microservices
Related to this discussion are microservices, which can be described as a suite of 
independently deployable, small, modular services. Each service runs a unique process and 
communicates through a well-defined, lightweight mechanism to serve a business goal. 1

Pinterest, for example, could have the following microservices: user profile, follower, feed, search, photo upload, etc.

Service Discovery
Systems such as Consul, Etcd, and Zookeeper can help services find each other by keeping 
track of registered names, addresses, and ports. Health checks help verify service 
integrity and are often done using an HTTP endpoint. Both Consul and Etcd have a built in 
key-value store that can be useful for storing config values and other shared data.

Disadvantage(s): application layer
Adding an application layer with loosely coupled services requires a different approach from an architectural, operations, and process viewpoint (vs a monolithic system).
Microservices can add complexity in terms of deployments and operations.
Source(s) and further reading
Intro to architecting systems for scale
Crack the system design interview
Service oriented architecture
Introduction to Zookeeper
Here's what you need to know about building microservices



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
