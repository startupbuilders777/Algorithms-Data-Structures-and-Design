scrape this:
https://leetcode.com/discuss/interview-question/4366889/System-Design-100-topics-to-learn

SCRAPE THIS (GOOD ADVANCED SYSTEM DESIGN NOTES) THEORETICAL. 
https://tikv.org/deep-dive/consensus-algorithm/raft/

GOOD NOTES ON PATTERNS HERE SCRAPE THIS:
https://microservices.io/patterns/data/cqrs.html

SCARPE THIS BLOG, ITS REALLY GOOD, READ ALL ITS POSTS: 
best system design prep: 
https://systemdesign.one/live-comment-system-design/
^ THIS IS THE MAIN BLOG LEARN BOTEC FROM HERE. 


1 million requests/day = 12 requests/second

L1 and L2 caches: 1 ns, 10 ns
E.g.: They are usually built onto the microprocessor chip. Unless you work with hardware directly, you probably don’t need to worry about them.

RAM access: 100 ns
E.g.: It takes around 100 ns to read data from memory. Redis is an in-memory data store, so it takes about 100 ns to read data from Redis.

Send 1K bytes over 1 Gbps network: 10 us
E.g.: It takes around 10 us to send 1KB of data from Memcached through the network.

Read from SSD: 100 us
E.g.: RocksDB is a disk-based K/V store, so the read latency is around 100 us on SSD.

Database insert operation: 1 ms.
E.g.: Postgresql commit might take 1ms. The database needs to store the data, create the index, and flush logs. 
All these actions take time.

Send packet CA->Netherlands->CA: 100 ms
E.g.: If we have a long-distance Zoom call, the latency might be around 100 ms.

Retry/refresh internal: 1-10s
E.g: In a monitoring system, the refresh interval is usually set to 5~10 seconds (default value on Grafana).

 


The real-time Application Programming Interface (API) can be implemented for faster user experiences and instant delivery of live comments. 
The average time for a human to blink is 100 ms, and the average reaction time for a human is around 250 ms. Therefore, the actions performed within 250 ms are perceived as real-time or live 4, 5. An event-driven architecture can be used to build a real-time data platform. The general subscription models for an API are the following 6:

push-based (server-initiated)
pull-based (client-initiated)
The popular protocols for an event-driven API are the following

Protocol	Description	Use Cases	Subscription Model
Webhook	HTTP-based callback function that allows lightweight, event-driven infrequent communication between APIs	trigger automation workflows	push-based
WebSub	communication channel for frequent messages between web content publishers and subscribers based on HTTP webhooks	news aggregator platforms, stock exchanges, and air traffic networks	push-based
WebSockets	provides full-duplex communication channels over a single TCP connection with lower overhead than half-duplex alternatives such as HTTP polling	financial tickers, location-based apps, and chat solutions	pull-based
SSE	lightweight and subscribe-only protocol for event-driven data streams	live score updates	pull-based
MQTT	protocol for streaming data between devices with limited CPU power and low bandwidth networks	Internet of Things	pull-based

The client creates a regular HTTP long poll connection with the server with server-sent events (SSE). The server can push a continuous stream of data to the client on the same connection as events occur. The client doesn’t need to perform subsequent requests 7.


The only difference for SSE from a regular HTTP request is that the Accept header on the HTTP request holds the value text/event-stream. The EventSource interface is used by the client to receive and process server-sent events independently in text/event-stream format without closing the connection. All modern web browsers support the EventSource interface natively. The EventSource interface can be implemented on iOS and Android platforms with lightweight libraries 7, 8.

The predefined fields of an SSE connection are the following 6:

Field	Description
event	the event type defined by the server
data	the payload of the event
id	ID for each event
retry	the client attempts to reconnect with the server after a specific timeframe if the connection was closed
The following are the drawbacks of SSE 6:

the data format is restricted to transporting UTF-8 messages with no support for binary data
only up to six concurrent SSE connections can be opened per web browser on pre-HTTP/2 networks
The components in the system expose the Application Programming Interface (API) endpoints to the client through Representational State Transfer (REST). The description of HTTP Request headers is the following:

Header	Description
accept	type of content the client can understand
authorization	authorize your user account
content-encoding	compression type used by the data payload
method	HTTP Verb
content-type	type of data format (JSON or XML)
The description of HTTP Response headers is the following:

Header	Description
status code	shows if the request was successful
content-type	type of data format

How does the receiver subscribe to a specific live video?
The client must subscribe to a live video for viewing the live comments. The client executes an HTTP PUT request for subscribing to a live video. The PUT requests are idempotent. The PUT method is used instead of the GET method because the in-memory subscription store will be modified when a client subscribes to a live video.

/videos/:video-id/subscriptions
method: PUT
accept: text/event-stream
authorization: Bearer <JWT>
content-length: 20
content-type: application/json
content-encoding: gzip

{
  user_id: <int>
}
The accept: text/event-stream HTTP request header indicates that the client is waiting for an open connection to the event stream from the server to fetch live comments 9. The server responds with status code 200 OK on success.


status code: 200 OK
content-type: text/event-stream
The content-type: text/event-stream HTTP response header indicates that the server established an open connection to the event stream to dispatch events to the client. The response event stream contains the live comments.


id: 1
event: comment
data: {"awesome"}
data: {"hey there"}


Type of Data Store
The content of live comments contains only textual data and does not include any media files. A very fast and reliable database that not only persistently store data but also access the data quickly is a key feature for building the live comment service. The persistent storage of live comments is needed to retrieve the comments at a later point in time. In addition, the data kept in persistent storage can be used for auditing purposes 10. The relational database offers well-defined structures for comments, users, and videos. The relational database will be an optimal choice when the dataset is small. However, the relational database will be a suboptimal solution for live comment service due to the following scalability limitations 5:

the internal data structure adds delay to the data operations
complex queries are needed to reintegrate data because of data segregation
The creation of database indexes on video_id and created_at columns will improve the performance of the read operations at the expense of slow write operations. The NoSQL database such as Apache Cassandra can be used as persistent data storage for live comments due to the following reasons 5:

Log-structured merge-tree (LSM) based storage engine offers extremely high performance on writes
schemaless data model reduces the overhead of joining different tables
optimized natively for time series data
Apache Cassandra is not optimized for read operations due to the nature of the LSM-based storage engine. An in-memory database such as Redis can be used in combination with Apache Cassandra to improve the read performance and make the data storage layer scalable and performant for live comments. The geo-replication enabled Redis cache with a time to live (TTL) of 1 second can be added as a cache layer on top of Apache Cassandra to improve the read performance. In addition, live comments published on an extremely popular live video can be kept in the cache on network edges to improve the latency 10.

The sets data type in Redis can be used to efficiently store the live comments. The native deduplication logic of sets data type ensures that live comments are stored in memory without having an additional logic to prevent repeated live comments. The sorted set data type in Redis can be used for data integrity by maintaining the reverse chronological ordering of live comments. The sorted set data type can make use of the timestamp on live comments for sorting the live comments without implementing a custom sorting algorithm. The metadata of the publisher of a live comment can be stored in Redis hash data type for quick retrieval 4, 11, 5.

The receivers who are geographically located closer to the publisher of the live comment will see the live comment instantly while the receivers who are located on a different continent might see the live comment with a slight delay (lower than 250 ms) to favor availability and partition tolerance in the CAP theorem 10.



Capacity Planning
The rate of clients viewing the live comments is significantly higher than the rate of clients publishing the live comments. The calculated numbers are approximations. A few helpful tips on capacity planning during system design are the following:

1 million requests/day = 12 requests/second
round off the numbers for quicker calculations
write down the units while doing conversions

Traffic
The live comment service is a read-heavy sfystem. The Daily Active Users (DAU) count is 100 million. 
On average, the total number of daily live videos is 200 million. A live video receives 10 comments on average.

Description	Value
DAU (write)	2 billion
QPS (write)	12 thousand
read: write	100: 1
QPS (read)	1.2 million


SECONDS IN DAY -> 86400


Bandwidth
Ingress is the network traffic that enters the server when live comments are written. 
Egress is the network traffic that exits the servers when live comments are viewed. 
The network bandwidth is spread out across the globe depending on the location of the clients.

Description	Calculation	Total
Ingress	2 billion comments/day * 2 KB/comment * 10^(-5) day/sec	40 MB/sec
Egress	200 billion comments/day * 2 KB/comment * 10^(-5) day/sec	4 GB/sec

Memory
The in-memory subscription store keeps the client viewership associations.

Description	Calculation	Total
Subscription store	100 million users/day * 16 bytes/user	1.6 GB/day


spark ARCHITECTURE
https://www.slideshare.net/slideshow/apache-spark-architecture/54851426

ZOOKEEPER:
https://www.slideshare.net/slideshow/introduction-to-apache-zookeeper/16567274

HDFS & MAP REDUCE

CHUBBY

HBase


C*


Redis


time series db

KAFKA

PRESTO

APACHE PINOT

sys design doc:
https://docs.google.com/document/d/16wtG6ZsThlu_YkloeyX8pp2OEjVebure/edit
##########


SYSTEM DESIGN SPECIFICS

Complex system design requires knowledge of threads and how they can be used to wait until something is complete

Use couting semaphores, tasks, threads, jobs, pipelines, for we
CONSISTENCY LEVELS AND EVENTUALLY CONSISTENT. 

aUHTORIZATION, Authentication

Talk about different stakeholders of system, talk about sister teams, talk about analytics, and ML divisions during system design conversion
TALK about how each feature scales using NFR

ELASTIC SEARCH REINDEXING jobs
BUILD FASTERING INDEXING READ DOORDASH GUIDE ON ELASTIC SEARCH:

https://doordash.engineering/2021/07/14/open-source-search-indexing/

read doordash guide on cassandra:

https://doordash.engineering/2024/01/30/cassandra-unleashed-how-we-enhanced-cassandra-fleets-efficiency-and-performance/
https://doordash.engineering/2023/11/07/leveraging-flink-to-detect-user-sessions-and-engage-doordash-consumers-with-real-time-notifications/
https://doordash.engineering/2023/02/07/how-we-scaled-new-verticals-fulfillment-backend-with-cockroachdb/
https://doordash.engineering/2023/02/22/how-doordash-designed-a-successful-write-heavy-scalable-and-reliable-inventory-platform/


C* notes:
    Schema design is important
    How will you deal with hot spots? 

    Updates and deletes are expensive in C*
    Partition key is hashed, -> use hot spot free key. 
    Try to make it so you only read one partition at a time, when you query and not multiple partitions. 
    Clustering column is importaant for range queries. 


    Avoid large partitions -> detreimental to C* performance. long garbage collection pauses, increased read latencies, challenges in compaction! 
    Consider the implications of secondary indexes: Secondary indexes in Cassandra can be useful but come with trade-offs. 
    They can add overhead and may not always be efficient, '
    especially if the indexed columns have high cardinality or if the query patterns do not leverage the strengths of secondary indexes.

    TTL and tombstones management: Time-to-live, or TTL, is a powerful feature in Cassandra for managing data expiration. However, it’s important to understand how TTL and 
    the resulting tombstones affect performance. Improper handling of tombstones 
    can lead to performance degradation over time. If possible, avoid deletes.

    Update strategies: Understand how updates work in Cassandra. Because updates are essentially write operations, 
    they can lead to the creation of multiple versions of a row that need to be resolved at read time, which impacts performance.
    Design your update patterns to minimize such impacts. If possible, avoid updates

    Understanding consistency levels: In Cassandra, consistency levels range from ONE (where the operation requires confirmation from a single node) to ALL (where the operation needs acknowledgment from all replicas in the cluster). There are also levels like QUORUM (requiring a majority of the nodes) and LOCAL_QUORUM (a majority within the local data center). Each of these levels has its own implications on performance and data accuracy. You can learn more about those levels in the configurations here. 
    Performance vs. accuracy trade-off: Lower consistency levels like ONE can offer higher performance because they require fewer nodes to respond. However, they also carry a higher risk of data inconsistency. Higher levels like ALL ensure strong consistency but can significantly impact performance and availability, especially in a multi-datacenter setup.
    Impact on availability and fault tolerance: Higher consistency levels can also impact the availability of your application. For example, if you use a consistency level of ALL, and even one replica is down, the operation will fail. Therefore, it's important to balance the need for consistency with the potential for node failures and network issues.
    Dynamic adjustment based on use case: One strategy is to dynamically adjust consistency levels based on the criticality of the operation or the current state of the cluster. This approach requires a more sophisticated application logic but can optimize both performance and data accuracy.


    Compaction is a maintenance process in Cassandra that merges multiple SSTables, or  sorted string tables, into a single one. 
    Compaction is performed to reclaim space, improve read performance, clean up tombstones, and optimize disk I/O.

    Users should choose from three main strategies to trigger compaction in Cassandra users based on their use cases. Each strategy is optimized for different things: 

    Size-tiered compaction strategy, or STCS
        Trigger mechanism:
        The strategy monitors the size of SSTables. When a certain number reach roughly the same size, the compaction process is triggered for those SSTables. For example, if the system has a threshold set for four, when four SSTables reach a similar size they will be merged into one during the compaction process.
        When to use:
        Write-intensive workloads
        Consistent SSTable sizes
        Pros:
        Reduced write amplification
        Good writing performance
        Cons:
        Potential drop in read performance because of increased SSTable scans
        Merges older and newer data over time
        You must leave much larger spare disk to effectively run this compaction strategy
    Leveled compaction strategy, or LCS
        Trigger mechanism:
        Data is organized into levels. Level 0 (L0) is special and contains newly flushed or compacted SSTables. When the number of SSTables in L0 surpasses a specific threshold (for example 10 SSTables), these SSTables are compacted with the SSTables in Level 1 (L1). When L1 grows beyond its size limit, it gets compacted with L2, and so on.
        When to use:
        Read-intensive workloads
        Needing consistent read performance
        Disk space management is vital
        Pros:
        Predictable read performance because of fewer SSTables
        Efficient disk space utilization
        Cons:
        Increased write amplification
    TimeWindow compaction strategy, or TWCS
        Trigger mechanism:
        SSTables are grouped based on the data's timestamp, creating distinct time windows such as daily or hourly. When a time window expires — meaning we've moved to the next window — the SSTables within that expired window become candidates for compaction. Only SSTables within the same window are compacted together, ensuring temporal data locality.
        When to use:
        Time-series data or predictable lifecycle data
        TTL-based expirations
        Pros:
        Efficient time-series data handling
        Reduced read amplification for time-stamped queries
        Immutable older SSTables
        Cons:
        Not suitable for non-temporal workloads
        Potential space issues if data within a time window is vast and varies significantly between windows
        
        In our experience, unless you are strictly storing time series data with predefined TTL, LCS should be your default choice. 
        Even when your application is write-intensive, the extra disk space required by progressively large SSTables under STCS makes 
        this strategy unappealing. LCS is a no-brainer in read-intensive use cases.


        It’s easy to forget that each compaction strategy should have a different bloom filter cache size. When you switch between compaction strategies, 
        do not forget to adjust this  cache size accordingly. 


    To batch or not to batch? It’s a hard question
        In traditional relational databases, batching operations is a common technique to improve performance because it can reduce network round trips and streamline transaction management. However, when working with a distributed database like Cassandra, the batching approach, whether for reads or writes, requires careful consideration because of its unique architecture and data distribution methods.

        Batched writes: The trade-offs
        Cassandra, optimized for high write throughput, handles individual write operations efficiently across its distributed nodes. But batched writes, rather than improving performance, can introduce several challenges, such as:

        Increased load on coordinator nodes: Large batches can create bottlenecks at the coordinator node, which is responsible for managing the distribution of these write operations.
        Write amplification: Batching can lead to more data being written to disk than necessary, straining the I/O subsystem.
        Potential for latency and failures: Large batch operations might exceed timeout thresholds, leading to partial writes or the need for retries.
        Given these factors, we often find smaller, frequent batches or individual writes more effective, ensuring a more balanced load distribution and consistent performance.


        Batched reads: A different perspective
        Batched reads in Cassandra, or multi-get operations, involve fetching data from multiple rows or partitions. 
        While seemingly efficient, this approach comes with its own set of complications:

        Coordinator and network overhead: The coordinator node must query across multiple nodes, potentially increasing response times.
        Impact on large partitions: Large batched reads can lead to performance issues, especially from big partitions.
        Data locality and distribution: Batching can disrupt data locality, a key factor in Cassandra's performance, leading to slower operations.
        Risk of hotspots: Unevenly distributed batched reads can create hotspots, affecting load balancing across the cluster.
        To mitigate these issues, it can be more beneficial to work with targeted read operations that align with Cassandra’s strengths in handling distributed data.

    In our journey at DoorDash, we've learned that batching in Cassandra does not follow the conventional wisdom 
    of traditional RDBMS systems. Whether it's for reads or writes, each batched operation must be carefully evaluated in the 
    context of Cassandra’s distributed nature and data handling characteristics. By doing so, we've managed to optimize our 
    Cassandra use, achieving a balance between performance, reliability, and resource efficiency.


You can use elastic search to store a user index, which contains information about the user like geohash, etc. 

Then you can query for users around you using leastic search geohash index!
Inverted indexes are so useful to search for anything you want. 

REMBER YOU CAN LEVELRAGE THE CLIENTS PHONE, OR CLIENTS DEVIDE, AND CREATE A CACHE THERE AS WELL AS DO LOGIC ON THE CLIENT DEVIDE TO 
REDUCE LOAD ON THE BACKEND. 


Great design! Realized that keeping recent updates in a cache for a short TTL can actually work as a design pattern
 whenever eventual consistency becomes an issue.

BTW in reality, isn't it more likely that Tinder like apps will use a ML pipeline to feed a queue of recommendations? My first thought was to go in that direction (not sure what kind of traps I would run into though :)).


https://www.hellointerview.com/learn/system-design/answer-keys/tinder

Going through this guys stuff is agold mine ^^^^





Elastic search reindex job looks like this

CDC -> KAFKA -> assembler job takes id and calls api to get complete data -> kafka -> ES SINK flink job has a ES connector to sink a batch write to elastic search. 
etl -> kafka 
batch reads/writes!!

Fault tolerance -> talk about what happens when different parts fail and how it is addressed so that entire system is production ready. 
node failures, network issues, event dellays, etc. 

Compliance -> tlak about gdpr support + 
security.
data encryption for security!

CLOUDFRONT SIGNED URLS WITH BUILTIN AUTHENTICATED S3 links

if you use redis make sure to specify TTL and Cache eviction policy LRU 
Cassandra also has TTL -> and ssl table compression settings, etc... -> which favors reads/writes 

kafka, events, compression, binary, avro, etc. 


GRPC, 
Vectors for conflict resolution of writes... multileader replication and total ordering vs causal -> vector clocks, 
rEAD REPAIR FOR CASSANDRA + backgrpund thread for repair 

read repair, quorums, gossip protocols, ... ANTI ENTROPY PROTOCL
read+ write quorums > nodes 

Sloppy quorums and hinted handoff1
lamport clocks and arbitrary ordering to do Last writer wins write conflicts... or keep all data there and have custom application logic to deal with 
multiple writes or use CRDTS so that all writes can go in the same way, and get processed properly idempotently even if there are conflicts, 


CDC is change data capture and CDC is a mechanism that reads data changes from the databse and applies the changes to another data system 
one comon solution is debezium, it uses a source connector to read changes from a db and applies them to cache solutions such as Redis. 

maintain consistency between SQL db and cache using CDC. 

SAGA --> a saga is a sequence of local transactoions. eah transaction updates and publishes messages to trigger the next trasnaction setep. 
if a step fails, the saga executes a ompensating tx to undo the cahnges that were made by precedeing transactions. 2PC works asa s ingle commit to 
perofrm acid tx, while saga consits of multiple steps and relies on eventual consistency. 

pessimistic locking vs optimizing locking with version numbers v1/v2 vs database constraints -> these are the 3 ways to deal with high contention data!!

learn ticketmaster for distributed locks 

rmbr to talb about pagination in api endpoints whne youdescibe them 


SERIALIZED and desearialize data to binary to save on networking cost + batching 

dont forget to talk about alerting, monitoring, promql, processing, oncall, tasks, handoff, kaka queue size alert make sure its small,
cpu/memory, slack channel where alerts can be seen, 

Dont forget to talk about analytics team, machine learning team, database team -> include them in overall discussion of design and use cases. 

whne you are done main design, add more funcitonal requirements and non funciton and explain how all the non functionals are being met. 

talk about kafka  +cache somewhere, redis pub/sub, cdns, APIM gateway, design principles and domain driven design databases, 
ACID databases VS BASE, CAP theorem for each data base, and 
RMBR TO go over apis before startign your high level design, and go over numbers after FR/NFR, or say you will do numbers in a bit. 



make sure to always be talking, DO NOT BE HAND WAYY LIKE IN THE AIRBNB INTERVIEW, PLEASE RIGHT OUT ALL THE DATABASE TABLE NAMES, AND COLUMNS 

AND KEEP ADDING DETAIL TOT HE MAIN DESIGN!, WRITE OUT ALL THE API CALL REST/POST/GRPC, SERVER SIDE EVENTS VS WEBSOCKETS, choose SSE

talk about how to stream downloads as multipart files, 

Kafka uses topic partitioning to improve scalability. In partitioning a topic, Kafka breaks 
it into fractions and stores each of them in different nodes of its distributed system. That 
number of fractions is determined by us or by the cluster default configurations.

Kafka guarantees the order of the events within the same topic partition. However, by default, 
it does not guarantee the order of events across all partitions.

kafka -> toppics -> each topic has partitions, use partition key here -> then partitions are replicated, and then kafka brokers exist .
brokers hold partitions, scale topic capcity by expanding the number of partitions hwihc is sharding. 
servers of kafka are brokers. EACH PARTITION HAS OFFSET PER CONSUMER .

the choice of batching messages is a classic tradeoff between throughput and latency 

each consumer group has a coordinator node in kafka? which is also the broker in kafka. 
coordinates the consumer group. 
consumer fetches based on committed offset. consumer commits the offset to the broker. 
there is also consumer rebalancing by broker. 
coordinator receives heartbeats from consumers. 

rebalance is done so that if consumer dies, other consumers can read partitions of topic to ensure all mesages are read. 
there is a leader in the consumer group generates a new partitions dispatch plan. 


REDIS HAS CRDTs and CRDB -> conflict replicated data type. 


Producers can produces messages to kafka with different ACK settings ACK=all means all partition replicas must have replicated message and 
all replicas and leader are in sync, ack=1 means only leader needs to commit and not wait for in sync replicas, ack=0 means producer doesnt 
care when partition has ackowledged receienved message and can lead to data loss, ack=1 can also lead to data loss a tradeoff between 
latency and durability for kafka.

if consumer in consumer group fails, or leaves, the broker node acting as coordination service will help reestablish the consumer group with new 
consumer group leader to do partition dispatch plan.

replicas should not be in the same broker node, incase broker node fails, then we cannot access this partition at all which is very bad!!

dead letter queue for retrying messages, to not block incoming messages. 

Use a storage system like HDFS or object store to store historical kafka event data in cold storage.



IF THE number of consumers of a grpip is larger than the number of partitions of a topic, 
some consumers will not get data from this topic. 
parallelism max here. 

point to points vs publish subscribe!!

Coordination service:
service discovery -> which kafka brokers are still alive. 
leader election -> one of the brokers is selected as the active controller -> only one active controller in the cluster and is responsible
        for assigning partitions. 
        zookeeper can be used for leader election here. or etcd. 
    






GOSSIP PROTOCOL VS ZOOKEEPER CONSENSUS BTW

KAFKA, topics, flink, figure out which kafka keys to use to leveraged keyed streams in flink and process events per key to reduce hotspots
and also figure out how to maximize parallelism both in flink and kafka based on number of partitions of the kafka topic, and increaseing 
the amount fo consumers in the consumer group (amd flink maps to one consumer group). 

GEOHASH vs QUAD TREE SERVER WHICH BUILDS QUAD TREE FROM READING IN S3. 
all of redis features, zookeeper... PAXOS, RAFT, 2 PHASE COMMIT, 2 PHASE LOCKING, SERIALIZABLE TRANSACTIONS, SERIALIZABLE SNAPSHOT, 
DISTRIButed transactions, ... LINEARIZABILITY.

Hot strage vs cold storage considerations

think about if you need a reconcilation job using snowflake and cdc to improve performacne

think about if you need kafka to update redis cache


USE THE CLIENT TO make it easier for backend server to process such as form zoom, client encodes all 3 signals and sends, instead of 
server doing all the work

also client can batch calls before sending to optimize performance instead of server dealing with all events!(reduces write qps for us )

cdn and point of presence servers which are edge servers to clients


instead of using a database you can just store parquet files in object storage, and have them sorted by geohash or howeever you want to access
and heavily cache files. this way dont have to pay costs of database. 

Okay, a bit about your use case: Parquet is the better option for you. This is why:

You aggregate raw data on really large and not splitted datasets
Your Spark ML Job sounds like a scheduled, not long-running job. (onces a week, day?)
This fits more in the use cases of Parquet. Parquet is a solution for ad-hoc analysis, filter analysis stuff. Parquet is really nice if you need to run a query 1 or 2 times a month. Parquet is also a nice solution if a marketing guy wants to know one thing and the response time is not so important. Simply and short:

Use Cassandra if you know the queries.
Use Cassandra if a query will be used in a daily business
Use Cassandra if Realtime matters (I talk about a maximum of 30 seconds latency, from, customer makes an action and I can see the result in my dashboard)

Use Parquet if Realtime doesn't matter

Use Parquet if the query will not perform 100x a day.
Use Parquet if you want to do batch processing stuff

Parquet files support complex nested data structures in a flat format and offer multiple compression options.

Parquet is broadly accessible. It supports multiple coding languages, including Java, C++, and Python, to reach a broad audience. This makes it usable in nearly any big data setting. As it’s open source, it avoids vendor lock-in.

Parquet is also self-describing. It contains metadata that includes file schema and structure. You can use this to separate different services for writing, storing, and reading Parquet files.  

Parquet files are composed of row groups, header and footer. Each row group contains data from the same columns. The same columns are stored together in each row group:

Ideal Parquet format use cases

Storing big data of any kind (structured data tables, images, videos, documents).
Ideal for services such as AWS Athena and Amazon Redshift Spectrum, which are serverless, interactive technologies.
A good fit for Snowflake as it supports extremely efficient compression and encoding schemes. 
When your full dataset has many columns, but you only need to access a subset.
When you want multiple services to consume the same data from object storage.
When you’re largely or wholly dependent on Spark.

Ideal ORC format use cases

When reads constitute a significantly higher volume than writes.
When you rely on Hive.
When compression flexibility/options are key.


Ideal Avro format use cases

Write-heavy operations (such as ingestion into a data lake) due to serialized row-based storage. 
When writing speed with schema evolution (adaptability to change in metadata) is critical.


S3 is not built "on top of" or "in line with" HDFS. HDFS and Amazon S3 are both ways to store lots of data, like giant digital filing cabinets. HDFS is like a big file cabinet where many people can work together on the same project, while S3 is like a really big, easy-to-access storage room where you can keep lots of different things. HDFS is used when people want to do complicated things with big sets of data, like analyzing patterns or making predictions, while S3 is used when you need a safe and easily accessible place to keep all your digital stuff. They both store data, but they're designed for different purposes and work in different ways.


As per Spark documentation, Spark can run without Hadoop.

You may run it as a Standalone mode without any resource manager.

But if you want to run in multi-node setup, you need a resource manager like YARN or Mesos and a distributed file system like HDFS,S3 etc.



discss about

hotspots

zookeeper and consistent hash ring to maintain a large list of redis pub/sub servers and determine which redis/pub server channel lives on 
using the consistent hash ring saved in zookeeper so websocket servers know which server + channel tO SUBscribe toooo

cASSANDRA what is partition key and what is clustering key know these things!

KAPPA vs lambda architecture

List of unique algorithms and write their pseudo code if you want to. 


active/passive failover vs active active failover in case redis fails there is an active standby

redis has active active also has geohash

Think about multileader replication vs leaderless vs single leader and how this affects databases and how you can shard database, 
and put db in different data centers in different geo zones so that writes can come from specific geozone go to that data center. 

Redis is also good for distributed locking by setting keys in redis for reservation systems like ticketmaster. 
can be done with redlock or can be done with 
A very simple distributed lock with a timeout might use the atomic increment (INCR) with a TTL. 
When we want to try to acquire the lock, we run INCR. If the response is 1 (i.e. we own the lock), 
we proceed. If the response is > 1 (i.e. someone else has the lock), we wait and retry again later. 
When we're done with the lock, we can DEL the key so that other proceesses can make use of it.


Redis for Leaderboards
Redis' sorted sets maintain ordered data which can be queried in log time which make them appropriate for leaderboard applications. 
The high write throughput and low read latency make this especially useful for scaled applications where something like a SQL DB will start to struggle.

In Tweet Search we have a need to find the tweets which contain a given keyword (e.g. "tiger") 
which have the most likes (e.g. "Tiger Woods made an appearance..." @ 500 likes).

We can use Redis' sorted sets to maintain a list of the top liked tweets for a given keyword. Periodically, we can remove low-ranked tweets to save space.

ZADD tiger_tweets 500 "SomeId1" # Add the Tiger woods tweet
ZADD tiger_tweets 1 "SomeId2" # Add some tweet about zoo tigers
ZREMRANGEBYRANK tiger_tweets 0 -5 # Remove all but the top 5 tweets


Redis for Rate Limiting
As a data structure server, implementing a wide variety of rate limiting algorithms is possible. A common algorithm is a 
fixed-window rate limiter where we guarantee that the number of requests does not exceed N over some fixed window of time W.

Implementation of this in Redis is simple. When a request comes in, we increment (INCR) the key for our rate limiter and 
check the response. If the response is greater than N, we wait. If it's less than N, we can proceed. We call EXPIRE on 
our key so that after time period W, the value is reset.


Redis for Proximity Search
Redis natively supports geospatial indexes with commands like GEOADD and GEORADIUS. The basic commands are simple:

GEOADD key longitude latitude member # Adds "member" to the index at key "key"
GEORADIUS key longitude latitude radius # Searches the index at key "key" at specified position and radius
The search command, in this instance, runs in O(N+log(M)) time where N is the number of elements in the radius and M is the number of members in our index.

Redis can be replicated to have main and Secondary

or you can have a redis cluster which handles different key ranges

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
		  - Amount of data to store on disk/SSD

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

############

Long polling vs client polling vs subscribe to webhook vs event based kafka stream vs skinny payloads vs atleast once vs large water mark 

saga pattern:

Advantages of Saga
This event-driven microservices architecture offers a highly scalable or loosely linked consistency with the use of transactions.
Using NoSQL databases without 2PC support is possible to offer surface-through transactions.


Choreography is like having a choreographer set all the rules. Then the dancers on stage (the microservices) interact according to them. Service choreography 
describes this exchange of messages and the rules by which the microservices interact.
Orchestration is different. The orchestrator acts as a center of authority. It is responsible for invoking and combining the services. It
 38
describes the interactions between all the participating services. It is just like a conductor leading the musicians in a musical symphony. The orchestration pattern also includes the transaction management among different services.
The benefits of orchestration:
1. Reliability - orchestration has built-in transaction management and error handling, while choreography is point-to-point communications and the fault tolerance scenarios are much more complicated.
2. Scalability - when adding a new service into orchestration, only the orchestrator needs to modify the interaction rules, while in choreography all the interacting services need to be modified.
Some limitations of orchestration:
1. Performance - all the services talk via a centralized orchestrator, so latency is higher than it is with choreography. Also, the throughput is bound to the capacity of the orchestrator.
2. Single point of failure - if the orchestrator goes down, no services can talk to each other. To mitigate this, the orchestrator must be highly available.
Real-world use case: Netflix Conductor is a microservice orchestrator and you can read more details on the orchestrator design.

3.2 Aggregator Pattern
3.3 API Gateway Design Pattern

3.4 Strangler Pattern

3.5 Event Sourcing Design Pattern

3.6 Command Query Responsibility Segregation (CQRS)
Basically, the command component in the command query responsibility segregation microservice 
design pattern implies that it will be in charge of creating, deleting, and updating the statements 
of the application while on the other hand, the query section will be reading all the statements.

Advantages of Command Query Responsibility Segregation
CQRS enables the availability of data with better speed.
With this pattern, data reading in event-driven Microservices is possible at a faster level.
It enables developers to scale both read and write systems independently.

Advantages of Decomposition Patterns
Decomposition pattern enables cohesive and loosely coupled approaches. 
It allows developers to decompose applications based on business capabilities or sub-domains of the applications.
Both business capabilities and sub-domains are relatively stable with this architecture.

Cache miss attack -> USE BLOOM FILTER TO ADDRESS THIS or store key = null in cache. 

Optimistic locking and saving db rows with higher version counts. 


CACHING STRATEGIES:
cache aside read Cache-Aside (Lazy Loading)
When your application needs to read data from the database, it checks the cache first to determine whether the data is available. If the data is available (a cache hit), the cached data is returned, and the response is issued to the caller.

while Read/Write Through provides a hands-free option for maintaining cache consistency. 
Write Behind/Write Back shines when optimizing for write-heavy workloads and can tolerate eventual consistency.

The main differences between read-through and cache-aside is that in a cache-aside strategy the application is responsible for fetching the data and populating the cache, while in a read-through setup, the logic is done by a library or some separate cache provider.

What is read-through caching?
Using Read-through & Write-through in Distributed Cache - NCache
Auto-refresh cache on expiration: Read-through allows the cache to automatically reload an object from the database when it expires. This means that your application does not have to hit the database in peak hours because the latest data is always in the cache.


read through 

write around (write to cache and db seperately. )
write back (write from cache to db once in a while)
write through (write to immedeitly to db from cache)


FANOUT ON READ VS FANOUT ON WRITE?

When to use OLAP vs. OLTP. Online analytical processing (OLAP) and online transaction processing (OLTP) are two different data 
processing systems designed for different purposes. 
OLAP is optimized for complex data analysis and reporting, while OLTP is optimized for transactional processing and real-time updates.

Common database categories include: 🔹
Relational
Columnar
Key-value
In-memory
Wide column
Time Series
Immutable ledger
Geospatial
Graph
Document
Text search
Blob

push vs pull based. 

analytics vs writes:

+1. Matt - look up OLTP and OLAP. Use one database to support fast transactional work (such as to support an app), 
and ETL data as required to a another database built to be reported off. As they are separate you don't get performance 
hits on one affecting the other. Moving data between the databases is a complex issue on it's own - depending on how 
much data there is to move, and how often; when people say they want "real-time" replication between sources what do 
they actually mean (you need to verify) because real-time to a computer is much faster than 'real-time' to a human.


2 Phase locking is a mechanism implemented within a single database instance to achieve serializeable isolation level. 
Serializeable transaction level is the strongest isolation where even with parallely executing transactions, 
the end result is same as if the transactions where executed serially. It works as follows:

Whenever the transaction wants to update an object/row, it must acquire a write/exclusive lock. When transactions 
wants to read an object/row, it must acquire a read/shared lock. Instead of releasing the lock immediately after each query, 
the locks must be held till the end of the transaction(commit or abort). So while the transaction is being executed, the number of locks 
held by the transaction expand/grow. (Read/write lock behavior is similar to any other reader/writer locking mechanisms, so not discussing here)

At the end of the transaction, the locks are released and number of locks held by the transactions shrinks.

Since the locks are acquired in one phase and released in another phase i.e., there are no lock releases in acquire phase and no new lock acquire in 
release phase, this is called 2 phase locking.

2 phase commit is an algorithm for implementing distributed transaction across multiple database instances to ensure all nodes either commit or abort the transaction.

It works by having coordinator(could be a separate service or library within the application initiating the transaction) issue two requests - PREPARE to all nodes in phase 1 and COMMIT(if all nodes returned OK in PREPARE phase) or ABORT(if any node returned NOT OK in PREPARE PHASE) to all nodes in phase 2.

TLDR:

2 phase locking - for serializable isolation within a single database instance

2 phase commit - atomic commit across multiple nodes of a distributed database/datastores


47

I am trying to understand the difference between paxos and two phase commit as means to reach consensus among multiple machines. Two phase commit and three phase commit is very easy to understand. It also seems that 3PC solves the failure problem that would block in 2PC. So I don't really understand what Paxos is solving. Can anyone illuminate me about what problem does Paxos exactly solve?

2PC blocks if the transaction manager fails, requiring human intervention to restart. 3PC algorithms (there are several such algorithms) try to fix 2PC by electing a new transaction manager when the original manager fails.

Paxos does not block as long as a majority of processes (managers) are correct. Paxos actually solves the more general problem of consensus, hence, it can be used to implement transaction commit as well. In comparison to 2PC it requires more messages, but it is resilient to manager failures. In comparison to most 3PC algorithms, Paxos renders a simpler, more efficient algorithm (minimal message delay), and has been proved to be correct.

Gray and Lamport compare 2PC and Paxos in an excellent paper titled "Consensus on Transaction Commit".

(In the answer of peter, I think he is mixing 2PC with 2PL (two-phase locking).)

And then three is Raft which is a more light-weight version of Paxos. There are a lot of open-source systems using raft right now. Such as Etcd, Consul, Cockroachdb, etc.

2-PC is the most traditional transaction commit protocol and powers the core of atomicity of transactions. But it is blocking in nature, i.e. if the transaction manager/coordinator fails in between, it will cause the protocol to block and no process will be aware of it. It requires manual intervention to repair the coordinator.

While Paxos being a distributed consensus protocol has multiple such coordinators and if a majority of the coordinators agree to the transaction completion, it becomes a successful atomic transaction.


Problem
When data needs to be atomically stored on multiple cluster nodes, nodes cannot make the data accessible to clients until the decision of other cluster nodes is known. Each node needs to know if other nodes successfully stored the data or if they failed.

Solution
The essence of two-phase commit, unsurprisingly, is that it carries out an update in two phases:

The prepare phase asks each node if it can promise to carry out the update.
The commit phase actually carries it out.

As part of the prepare phase, each node participating in the transaction acquires whatever it needs to assure that it will be able to do the commit in the second phase—for example, any locks that are required. Once each node is able to ensure it can commit in the second phase, it lets the coordinator know, promising the coordinator that it can and will commit in the second phase. If any node is unable to make that promise, then the coordinator tells all nodes to roll back, releasing any locks they have, and the transaction is aborted. Only if all the participants agree to go ahead does the second phase commence—at which point it's expected they will all successfully update. It is crucial for each participant to ensure the durability of their decisions using pattern like Write-Ahead Log. This means that even if a node crashes and subsequently restarts, it should be capable of completing the protocol without any issues.



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


Write-through#
Using the write-through policy, data is written to the cache and the backing store location at the same time. The significance here is not the order in which it happens or whether it happens in parallel. The significance is that I/O completion is only confirmed once the data has been written to both places.

def write_through(cache, backing_store, datum):
    cache.write(datum)
    backing_store.write(datum)
Advantage: Ensures fast retrieval while making sure the data is in the backing store and is not lost in case the cache is disrupted.

Disadvantage: Writing data will experience latency as you have to write to two places every time.

What is it good for?#
The write-through policy is good for applications that write and then re-read data frequently. This will result in slightly higher write latency but low read latency. So, it’s ok to spend a bit longer writing once, but then benefit from reading frequently with low latency.

Write-around#
Using the write-around policy, data is written only to the backing store without writing to the cache. So, I/O completion is confirmed as soon as the data is written to the backing store.

def write_around(backing_store, datum):
    backing_store.write(datum)
Advantage: Good for not flooding the cache with data that may not subsequently be re-read.

Disadvsntage: Reading recently written data will result in a cache miss (and so a higher latency) because the data can only be read from the slower backing store.

What is it good for?#
The write-around policy is good for applications that don’t frequently re-read recently written data. This will result in lower write latency but higher read latency which is a acceptable trade-off for these scenarios.

Write-back#
Using the write-back policy, data is written to the cache and Then I/O completion is confirmed. The data is then typically also written to the backing store in the background but the completion confirmation is not blocked on that.

def write_back(cache, datum):
    cache.write(datum)
    # Maybe kick-off writing to backing store asynchronously, but don't wait for it.
Advantage: Low latency and high throughput for write-intensive applications.

Disadvantage: There is data availability risk because the cache could fail (and so suffer from data loss) before the data is persisted to the backing store. This result in the data being lost.

What is it good for?#
The write-back policy is the best performer for mixed workloads as both read and write I/O have similar response time levels. In reality, you can add resiliency (e.g. by duplicating writes) to reduce the likelihood of data loss.

Which one should I use?#
If this post is all you know about caching policies then you’ll need to do more research. The post covered three basic caching policies at a _very high level just to give you a basic understanding and to spark an interest. In practice, there are many other (often hybrid) policies and lots of subtle nuggets to consider when implementing them.

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

 
+SYSTEM DESIGN STUDY GUIDE DDIA: https://docs.google.com/document/d/1cHB2rks07RAo0cEEeONZjwGHsVI5GULEbEZLB-OGgxs/edit
+
+
+SYSTEM DESIGN BIG CHEATSHEET:https://docs.google.com/document/d/16wtG6ZsThlu_YkloeyX8pp2OEjVebure/edit
+what is alex yu?
+ALEX YU INSIDE SYSTEM DESIGN INTERVIEW:
+
+Grab his second book too:
+ALEX YU INSIDE SYSTEM DESIGN INTERVIEW 2:
+
+https://www.bing.com/search?q=alex+yu+system+design&cvid=ed6dd359e5774b4797aec2dc7dccffa0&aqs=edge.0.0j69i57.1904j0j1&pglt=43&FORM=ANNTA1&PC=U531
+AS COOL AS DDIA
+
+Watch Code Karls Videos too: https://www.youtube.com/watch?v=YyOXt2MEkv4 -> he has a bunch of good example system design sessions
+-> Code karl also has descriptions here: https://www.codekarle.com/system-design/Uber-system-design.html for each vid
+https://www.youtube.com/watch?v=YyOXt2MEkv4 -> Airbnb System Design | Booking.com System Design | System Design Interview Question
+
+
+Another dope system design channel: https://www.youtube.com/c/ThinkSoftware/videos 
+
+Cool blog system desing very good: http://highscalability.com/blog/category/example
+
+Read this other system design patterns checklist:
+https://microservices.io/patterns/data/saga.html
+
+Application architecture patterns
+
+Monolithic architecture
+Microservice architecture
+Decomposition
+
+Decompose by business capability
+Decompose by subdomain
+Self-contained Servicenew
+Service per teamnew
+Refactoring to microservicesnew
+
+Strangler Application
+Anti-corruption layer
+Data management
+
+Database per Service
+Shared database
+Saga
+API Composition
+CQRS
+Domain event
+Event sourcing
+Transactional messaging
+
+Transactional outbox
+Transaction log tailing
+Polling publisher
+Testing
+
+Service Component Test
+Consumer-driven contract test
+Consumer-side contract test
+Deployment patterns
+
+Multiple service instances per host
+Service instance per host
+Service instance per VM
+Service instance per Container
+Serverless deployment
+Service deployment platform
+Cross cutting concerns
+
+Microservice chassis
+Service Template
+Externalized configuration
+Communication style
+
+Remote Procedure Invocation
+Messaging
+Domain-specific protocol
+Idempotent Consumer
+External API
+
+API gateway
+Backend for front-end
+Service discovery
+
+Client-side discovery
+Server-side discovery
+Service registry
+Self registration
+3rd party registration
+Reliability
+
+Circuit Breaker
+Security
+
+Access Token
+Observability
+
+Log aggregation
+Application metrics
+Audit logging
+Distributed tracing
+Exception tracking
+Health check API
+Log deployments and changes
+UI patterns
+
+Server-side page fragment composition
+Client-side UI composition
+


REDIS------
Some of the most fundamental data structures supported by Redis:

Strings
Hashes (Objects)
Lists
Sets
Sorted Sets (Priority Queues)
Bloom Filters
Geospatial Indexes
Time Series

In addition to simple data structures, Redis also supports different communication patterns like Pub/Sub and Streams, partially standing in for more complex setups like Apache Kafka or Amazon's Simple Notification Service.

The core structure underneath Redis is a key-value store. All data structures stored in Redis are stored in keys: whether those be simple like a string or complex like a sorted set of bloom filter.

Commands
Redis' wire protocol is a custom query language comprised of simple strings which are used for all functionality of Redis. The CLI is really simple, you can literally connect to a Redis instance and run these commands from the CLI.

SET foo 1  
GET foo     # Returns 1
INCR foo    # Returns 2
XADD mystream * name Sara surname OConnor # Adds an item to a stream

Each node maintains some awareness of other nodes via a gossip protocol so, in limited instances, if you request a key from the wrong node you can be redirected to the correct node. But Redis' emphasis is on performance so hitting the correct endpoint first is a priority.

Compared to most databases, Redis clusters are surprisingly basic (and thus, have some pretty severe limitations on what they can do). Rather than solving scalability problems for you, Redis can be thought of as providing you some basic primitives on which you can solve them. As an example, with few exceptions, Redis expects all the data for a given request to be on a single node! Choosing how to structure your keys is how you scale Redis.

Redis as a Distributed Lock
Another common use of Redis in system design settings is as a distributed lock. Occasionally we have data in our system and we need to maintain consistency during updates (e.g. the very common Design Ticketmaster system design question), or we need to make sure multiple people aren't performing an action at the same time (e.g. Design Uber).

Most databases (including Redis) will offer some consistency guarantees. If your core database can provide consistency, don't rely on a distributed lock which may introduce extra complexity and issues. Your interviewer will likely ask you to think through the edge cases in order to make sure you really understand the concept.

A very simple distributed lock with a timeout might use the atomic increment (INCR) with a TTL. When we want to try to acquire the lock, we run INCR. If the response is 1 (i.e. we own the lock), we proceed. If the response is > 1 (i.e. someone else has the lock), we wait and retry again later. When we're done with the lock, we can DEL the key so that other proceesses can make use of it.

More sophisticated locks in Redis can use the Redlock algorithm.

Redis for Leaderboards
Redis' sorted sets maintain ordered data which can be queried in log time which make them appropriate for leaderboard applications. The high write throughput and low read latency make this especially useful for scaled applications where something like a SQL DB will start to struggle.

In Tweet Search we have a need to find the tweets which contain a given keyword (e.g. "tiger") which have the most likes (e.g. "Tiger Woods made an appearance..." @ 500 likes).

We can use Redis' sorted sets to maintain a list of the top liked tweets for a given keyword. Periodically, we can remove low-ranked tweets to save space.

ZADD tiger_tweets 500 "SomeId1" # Add the Tiger woods tweet
ZADD tiger_tweets 1 "SomeId2" # Add some tweet about zoo tigers
ZREMRANGEBYRANK tiger_tweets 0 -5 # Remove all but the top 5 tweets

Redis for Rate Limiting
As a data structure server, implementing a wide variety of rate limiting algorithms is possible. A common algorithm is a fixed-window rate limiter where we guarantee that the number of requests does not exceed N over some fixed window of time W.

Implementation of this in Redis is simple. When a request comes in, we increment (INCR) the key for our rate limiter and check the response. If the response is greater than N, we wait. If it's less than N, we can proceed. We call EXPIRE on our key so that after time period W, the value is reset.

Redis for Proximity Search
Redis natively supports geospatial indexes with commands like GEOADD and GEORADIUS. The basic commands are simple:

GEOADD key longitude latitude member # Adds "member" to the index at key "key"
GEORADIUS key longitude latitude radius # Searches the index at key "key" at specified position and radius



Reliability
The presence database (Redis) should not lose the current status of the clients on a node failure. The following methods can be used to persist Redis data on persistent storage such as solid-state disk (SSD) 13, 14:

Redis Database (RDB) persistence performs point-in-time snapshots of the dataset at periodic intervals
Append Only File (AOF) persistence logs every write operation on the server for fault-tolerance
The RDB method is optimal for disaster recovery. However, there is a risk of data loss on unpredictable node failure because the snapshots are taken periodically. The AOF method is relatively more durable through an append-only log at the expense of larger storage needs. The general rule of thumb for improved reliability with Redis is to use both RDB and AOF persistence methods simultaneously 13.

Latency
The network hops in the presence platform are very few because the client SSE connections on the real-time platform are reused for the implementation of the presence feature. On top of that, the pipelining feature in Redis can be used to batch the query operations on the presence database for reducing the round-trip time (RTT) 15.


The sets data type in Redis is an unordered collection of unique members with no duplicates. The sets data type can be used to store the presence status of the clients at the expense of not showing the last active timestamp of the client. The user IDs of the connections of a particular client can be stored in a set named connections and the user IDs of every online user on the platform can be stored in a set named online.

The sets data type in Redis supports intersection operation between multiple sets. The intersection operation between the set online and set connections can be performed to identify the list of connections of a particular client, who is currently online. The following Redis set commands can be useful to prototype the presence platform 5, 6:

Command	Description
SADD	add the user to the online set
SISMEMBER	check if the user is online
SREM	remove the user from the online set
SCARD	fetch the total count of online users
SINTER	identify connections who are online
The set operations such as adding, removing, or checking whether an item is a set member take constant time complexity, O(1). The time complexity of the set intersection is O(n*m), where n is the cardinality of the smallest set and m is the number of sets. Alternatively, the bloom filter or cuckoo filter can be used to reduce memory usage at the expense of approximate results 5.


The INCR Redis command is used to increment the counter atomically. Redis can be deployed in the leader-follower replication topology for improved performance. The Redis proxy can be used as the sharding proxy to route the reads to the Redis follower instances and redirect the writes to the Redis leader instance. The hot standby configuration of Redis using replicas improves the fault tolerance of the distributed counter 16. The write-behind cache pattern can be used to asynchronously persist the counter in the relational database for improved durability.

Alternatively, Redis can be deployed in an active-active topology for low latency and high availability. Instead of aggregating the count by querying every shard on read operations, the shards can communicate with each other using the gossip protocol to prevent the read operation overhead. However, this approach will result in unnecessary bandwidth usage and increased storage needs 5.

The count update operations must be idempotent for conflict-free count synchronization across multiple data centers. The drawback of using native data types in Redis is that the count update operations are not idempotent and it is also non-trivial to check whether a command execution on Redis was successful during a network failure, which might result in an inaccurate counter. As a workaround to network failures, the Lua script can be executed on the Redis server to store the user IDs or keep the user IDs in a Redis set data type 5. In conclusion, do not use native data types in Redis to create the distributed counter.

CRDT Distributed Counter
The distributed counter is a replicated integer. The primary benefit of the eventual consistency model is that the database will remain available despite network partitions. The eventual consistency model typically handles conflicts due to concurrent writes on the same item through the following methods 3, 13:

Conflict resolution	Description	Tradeoffs
last write wins (LWW)	timestamp-based synchronization	difficult to synchronize the system clocks
quorum-based eventual consistency	wait for an acknowledgment from a majority of the replicas	increased network latency
merge replication	merge service resolves the conflicts	prone to bugs and slow in real-time
conflict-free replicated data type (CRDT)	mathematical properties for data convergence	conflict resolution semantics are predefined and cannot be overridden
The Conflict-free Replicated Data Type (CRDT) is a potential option for implementing the distributed counter. The CRDTs are also known as Convergent Replicated Data Types and Commutative Replicated Data Types. The CRDT is a replicated data type that enables operations to always converge to a consistent state among all replicas nodes 17. The CRDT allows lock-free concurrent reads and writes to the distributed counter on any replica node 18. The CRDT internally uses mathematically proven rules for automatic and deterministic conflict resolution. The CRDT strictly employs the following mathematical properties 3, 13:

Property	Formula
commutative	a * b = b * a
associative	a * ( b * c ) = ( a * b ) * c
idempotence	a * a = a
The idempotence property in CRDT prevents duplicating items on replication 3, 13. The order of replication does not matter because the commutative property in CRDT prevents race conditions 18.

CRDT-based consistency is a popular approach for multi-leader support because the CRDT offers high throughput, low latency for reads and writes, and tolerates network partitions 13, 19, 20. The CRDT achieves high availability by relaxing consistency constraints 21, 22. The CRDT can even accept writes on the nodes that are disconnected from the cluster because the data updates will eventually get merged with other nodes when the network connection is re-established 17.

The internal implementation of CRDT is independent of the underlying network layer. For example, the communication layer can use gossip protocol for an efficient replication across CRDT replicas 23, 24. The CRDT replication exchanges not only the data but also the operations performed on the data including their ordering and causality. The merging technique in CRDT will execute the received operations on the data. The following are the benefits of using CRDT to build the distributed counter 18, 23, 22, 21:

offers local latency on read and write operations through multi-leader replication
enables automatic and deterministic conflict resolution
tolerant to network partitions
allow concurrent count updates without coordination between replica nodes
achieves eventual consistency through asynchronous data updates
An active-active architecture is a data-resilient architecture that enables data access over different data centers. An active-active architecture can be accomplished in production by leveraging the CRDT-based data types in Redis Enterprise 18, 19, 17. The CRDTs in Redis are based on an alternative implementation of most Redis commands and native data types. Some of the popular CRDTs used in the industry are the following 3, 13:

G-counters (Grow-only)
PN-counters (Positive-Negative)
G-sets (Grow-only sets)


G-Counter
The G-Counter (Grow-only) is an increment-only CRDT counter. The major limitations of the G-Counter are that the counter can only be incremented and the number of replicas must be known beforehand. The G-Counter will not be able to satisfy the elastic scalability requirements 23, 22.

PN-Counter
The PN-Counter (Positive-Negative) is a distributed counter that supports increment and decrement operations. The PN-Counter internally uses two count variables for the increments and decrements respectively 25. The PN-Counter supports read-your-writes (RYW) and monotonic reads for strong eventual consistency 22, 21. In addition, the PN-Counter supports an arbitrary number of replicas for elastic scalability 25. In summary, the PN-Counter can be used to build the distributed counter.

Handoff Counter
The Handoff Counter is a distributed counter, which is eventually consistent. The Handoff Counter is relatively more scalable than PN-Counter by preventing the identity explosion problem. The state of the G-Counter is linear to the number of independent replicas that incremented the counter. In other words, each replica in CRDT has a unique identity, and CRDTs work by keeping the identities of the participating replicas.

The set of identities kept on each replica grows over time and might hinder scalability. The Handoff Counter prevents the global propagation of identities and performs garbage collection of transient entries. The Handoff Counter assigns a tier number to each replica to promote a hierarchical structure for scalability 24. In summary, the Handoff Counter can also be used to build the distributed counter.

How does CRDT work?
The CRDT-based counter support count synchronization across multiple data centers with reduced storage, low latency, and high performance. The updates on the counter are stored on the CRDT database in the local data center and asynchronously replicated to the CRDT database in peer data centers 22, 26. The CRDT replicas will converge to a consistent state as time elapses 22, 21. Every count update operation is transferred to all replicas to achieve eventual consistency.

The commutative property of CRDT operations ensures count accuracy among replicas. The following set of guidelines can be followed to ensure a consistent user experience by taking the best advantage of the CRDT-based database 3:

keep the application service stateless
choose the correct CRDT data type
Figure 15: CRDT counter replicas across multiple data centersFigure 15: CRDT counter replicas across multiple data centers
Each replica will keep only its count and count update operations instead of the total count. A replica would never increment the count contained in another replica. The count updates on a particular replica are assumed to occur in a monotonic fashion. The vector clock is used to identify the causality of count update operations among CRDT replicas 18, 3, 13. The following is the rough data schema of the counter CRDT 27, 28, 22:

1
2
3
4
{ 
  "replica-1": { "increments": 2500, "decrements": 817 }, 
  "replica-2": { "increments": 21000, "decrements": 9919 } 
}
The total count is calculated by summing up all the count values. The total count would equal 2500 + 21000 - 817 - 9919.

The Redis keyspace notifications can be used instead of the Lua script for notifications at the expense of more CPU power and increased delay. The heartbeat signals must be replaced with HTTP events to publish Facebook likes or reactions through the real-time platform. The article on the real-time presence platform describes in-depth the handling of jittery client connections 30.

+
+----------------------------------
+
+
+SCRAPE THE SYSTEM DESIGN LEETCODE ALGORITHMS:
+https://github.com/resumejob/system-design-algorithms
+
+
+https://giters.com/chaitanyaphalak/distributed_systems_notes#networking
+https://cs.uwaterloo.ca/~tozsu/courses/cs454/notes.html
+http://anthony-zhang.me/University-Notes/CS454/CS454.html
+
+
+
+------------------------------------
+WATCH THE YOUTUBE GUY:
+Channel name:  System Design Interview
+
+----------------------------------
+SYSETM DESIGN TEMPLATE (https://leetcode.com/discuss/career/229177/My-System-Design-Template)
+
+
+
+-----------------------------
+System Design Interview - Step By Step Guide
+https://www.youtube.com/watch?v=bUHFg8CZFws&t=3524s&ab_channel=SystemDesignInterview
+
+Remember -> everythign is trade-off -> always discuss tradeoffs with interviewer with respect to business requeimetns. 
+
+
+Stages of interview: 
+Interviewer ask us for system that does counting for views on youtube, or likes on instagram, or facebook.
+But more often, problem stated in more general manner. So we have to nail down on what they want!!
+
+We may be asked for several metrics. Performance of diff applications. How many requests, errors, avg response time, etc,
+Or analyze data in realtime
+
+What does data analysis mean, who sends us data, who uses data, who uses results of data, is there a ML analytics team??
+ALWAYS ASK QUESTSIONS:
+
+The interviewer wants to you to deal with ambiguity, why because we want to know how you approach design problems in real life. 
+We need you to talk and bring things up. 
+
+Interviewees need requirement clarifications so we know exact functional and nonfunctional requirements and what exact technologies 
+we should use based on CAP theorem! -> because there are many solns to problem asked and we need to pick the best ones. 
+
+Ask youtube view count to someone, how do they solve -> based on what they are experts on? 
+Could we use distributed cache to count stuff, Kafak+spark? sql database, nosql database, batchprocessing (hadoop Map reduce), Cloud native processing (Kinesis)
+These options not equal, each has pros/cons/tradeoffs. 
+
+Ask the following questions for 4 categories:  Ask to clarify what we need. 
+
+Ask about users/customers, Ask about scale, ask about performance, finally ask about cost (budget constraints). 
+use view counting as example.
+
+USERS: 
+Who will use system? All youtubers or statistics for video owner, or is the user a machine learning model? 
+
+
+How will the system be used? by marketing to generate monthly reports so not retriveved often? 
+                             do we need it in realtime or daily because it is used in realtime recommendation service? 
+                             HOW WILL THE SYSTEM BE USED? WHAT DATA SHOULD WE STORE? 
+
+SCALE: 
+How many read queries per second
+how much data is queried per request?
+how many video veiwers are processed per second
+can there be spikes in traffic? 
+The interviewer will help use deifne these numbers, or wecan assume reasonable values. 
+
+
+PERFORAMNCE (EVALUATE diff design options);
+can we count views several hours later than when it happens (then we can use batch procesing)
+
+What is expected write-to-read delay? Can we use batch processing or stream processing
+What is expected P99 latency for read queries.  (how fast should data be retrieved from system)
+-> if p99 matters, data should already be aggregated. 
+-> if interviwer says response time must be as small as possible, we must count views when we write data, and 
+    we should do minimal/no counting when we read data. 
+
+
+
+Coost (evaluate tech stack): 
+Should the design minimized the cost of deployment
+If we wanat to minimize developement cost, use well regarded open source frameworks. 
+
+Should the design minimize the cost of maintence. -> if future maintain cost is a concern, consider public cloud services for our design.
+
+IF YOU DONT ASK QUESTSION -> Interviewr will think you are a junion engineer. ALWAYS ASK!! TO INCREASE YOUR LEVELING!!
+You better spend additional 5 minutes clarifying requirements and scope, how data gets in and out of system VS solving a different,
+more complex problem than the interveiwer actually asked. 
+
+
+CLARIFY THE REQUIREMENTS SO 
+
+
+
+We want to be able to define functional and nonfunctional requirements (fast, fault tolerant, secure) after asking our questions. 
+
+Functional requriements API write down on whiteboard.:
+Write out the api!!
+
+This system has to count video view events:
+countViewEvent(videoId, ) OR
+
+countView(videoId, eventType(such as view like, share), function(such as count, sum, average) ) -> we can generalize parameters in api like this
+
+countView(videoId, eventType(such as view like, share), function(such as count, sum, average) ) -> can support total watch time, avg view duration, etc.
+processEvents(listOfEvents) -> process a list of events at once, instead of 1 event. 
+
+
+The system has to return video views count for a time period:
+
+getViewsCount(videoId, startTime, endTime)
+  ->
+getCount(videoId, , eventType, startTime, endTime)
+-> 
+getStats(videoId, , eventType, function, startTime, endTime)
+Name api in more generic way, as you add more parameters!
+
+CONTINUE TO GENERALIZE YOUR APIS as you define this. 
+
+
+NON FUNCTIONAL REQUIREMNTS: 
+
+interviewer wont tell us nonfunctional requirements -> try to think of business requirements and tradeoffs: 
+deal with big scale, and high performance -> we will have to handle following:
+
+5 CATEGORIES (FIRST 3 MOST IMPORTANT):
+Scalable (tens of thorusangs vide views per second)
+Highly Performant (few tens of milliseconds to return total views count for the video)
+highly available
+CONSISTENCY
+COST MINIMIZATION 
+
+NEXT STAGE OF SYSTEM DESING: HIGH LEVEL ARCHITECTURE: 
+
+we need a database, processing service (write view counts), and a query service (view view count on youtube),
+
+just draw boxes, and dont worry about it. Most forward one step at a time. 
+
+Start with something simple, than continue to aggregate. 
+
+WE NEED TO FIGURE OUT THE SMALLEST UNIT FIRST AND THEN build a system around it, in other words, the data!!
+Understand what data we need to store, and how we do it. 
+We need to define a data model!
+
+What we store: 
+
+Individdual events (every click) 
+-> Fast writes, later when we retreieve data, we can slice and dice how we want, and raw events allow use to recalaute stats as necessary 
+-> But we cant read data quickly, slow reads, it may also cost alot of money to store a lot of view events, costly for large scale. 
+ 
+
+Or we can store data on fly -> aggregate date (per minute) in realtime -> such as count. 
+-> Fast reads (dont need to aggregate), 
+-> data is ready for decision making in realtime -> can use data for recommendation service or trending service. 
+Drawbacks: can query only the way data was aggregated, requries data aggregation pieplein, hard or even impossible to fix errors
+
+
+Do we store raw events, or store aggragate data per minute -> WE NEED TO ASK INTERVIWER THIS, to help guide us 
+Ask interviwer about expected data delay, if it should be no more than a few minutes, we must aggregate on the fly, otherwise, 
+we can store raw events and do batch processing 
+
+Interviwer will let us know what she is interested in. And by the way combinging both apparoches makes alot of sense for many systems 
+we will store raw events, and because there are so many of them , we will store events for several days or weeks, then purge old data. 
+And we also calculate data in realtime, so stats available to user right away. 
+
+Doing both ways, means we have the best of both worlds!!
+But price to pay for all this flexibility -> system becomes more complex, and expensive -> good to talk about with interviwer. 
+
+NEXT QUESTION: 
+is WHERE WE STORE THE DATA!!!
+
+
+Both SQL and noSQL databases can scale and perform well.
+
+Evaluate databases based on non funcctional requirements. 
+How to scale writes?
+how to scale reads?
+How to make both writes and reads fast?
+How not to lose data in case of hardware faults and network partitions?
+how to achieve strong conssitency? what are the tradeoffs? 
+how to recover data in case of an outage?
+how to ensure data security?
+how to make it extensible for data model changes in the future?
+Where to run (cloud vs on-premise) data centers?
+
+
+SQL DATABASES!!
+simple when database on one machine, but what if one machine isnt enuff. 
+Data needs to be split on multiple machines, sharding aka horizontal partitioning, 
+Each shard holds subset of all data, several machines, services talk to machines must know which one databases exist and which to pick for queries, etc.
+
+Processing service writes, query service reads. 
+Need light proxy server which knows about all database machines and routes traffic to correct shards. both write and read services talk to cluster proxy only, dont need to know about each and every database machine anymore, 
+but cluster proxy has to know, moreover, proxy needs to know when some shards die, or become unavaiible to network partition, or when new shards added to database cluster, proxy needs to be aware of it. 
+proxy needs a new component, configuration service, configuration service matians health check connection to all shards so it knowas what database machines are available. 
+
+and cluster proxy needs to know when shards die, when shards added, etc. 
+Introduce a new component -> confgiuration service which has health monitor conection for database aka zookeeper. 
+
+AND INSTEAD OF CALLING DATABASE INSTACNCE DIRECTLY, WE CAN INTRODUCE ONE MORE PROXY, SHARD PROXY. 
+Introduce a shard proxy sit in front of database, shard proxy, can cache query results, monitor database instance health, and publish metrics, termiante queries that take too long to return data, and many more 
+Great setup helps us address several requirments before like scabaility and performance. 
+
+
+Availabilty needs to be addressed,what if shard dies, how to ensure data is not lost? 
+we need to replicate data, each existing shard, create read replicas, and a master shard. We call it read replica because all writes still go through master. 
+Also put read replicas in data center different from master shard. so if whole data center goes down, data still availabel. 
+
+so when Store data request comes, based on info from confgiuration service, cluster proxy sends data to a shard. And data is either syncrhonosly or async replicated to a corresponding read replica 
+And when retrieve data request comes, cluster may retreive either from master or read replica. 
+
+Ideas we just discussed is what youtube is using. 
+
+Great now we know how to scale sql databases. 
+But this solution doesnt seem simple right!
+
+We have all these proxies, configurations
+
+Read request, cluster proxy
+
+
+HOW ABOUT NOSQL(CASSANDRA?)
+
+in no sql, still split data into chunks, called shards, aka nodes. instead of leaders and followrs, we say each shard is equal. we no longer need congifuration service to monitor health of each shard. 
+instead of lets allow shards to talk to each other.
+And shards can talk to each other and exchange info about its state. 
+
+To reduce network load, we dont need each shard to talk to every other shard. every second shard may exchange info with a few other shards, no more than 3. 
+Quickl enuff state info about every node propogates throughout the cluster. This procedure is called a gossip protocol. 
+ok each node iin the clsuter knows about other nodes. And this is a big deal, rmbr that previously we used cluster proxy component to route request to particular shard. Cluster proxy only knew about all shards.
+But now every node knows about each other. So clients of our db do not have to call a speciail compoentn to route requests, clients can call any node in cluster. 
+nodes iteself decide where to forward requets. 
+
+
+Processing service makes a call to store views count for some video B. And lets say node 4 is selected to serve this requeist. 
+WWe can use a round robin algorithm to choose this initial node, or we may be smarter, and choose a node that is closted to the client in terms of network distance. Lets call this node 4 a coordinator node. The coordinator
+node needs to decide which node stores data for the requested video. We can  use consistent hashing algo to pick the node. As you may see node 1 should store the data for vid B. Coordinater node will make a call 
+to the node 1, and wait for the repsonse, nothing stops coordinator node to call 3 nodes for 3 replicas of data. Waiting for 3 responses too slow. 
+
+We should wait for 3 succss requests actually we wait for 2 success requests for speed. This process is called Quorum writes. 
+We consider right to be successful when 2 replication requests compelte. This is called QUORUM WRITES. SIMILAR TO QUORUM writes there 
+is quorum reads appraoch. Wehn query service retreive count for vid b, 
+coordinate node 4 will initiate several read requests in parallel. In theroy coordinate node may get diff respones from replica nodes? 
+Why? because some nodes could have beeen unavailable when write request happened.
+That node has stale data right now ,other 2 nodes has up to date data. Read quorum defines a minum number of nodes that have to agree on the response. 
+Cassandra uses version numbers to determine 
+staleness of data. And similar to sql database, we want to store copies of data across different data centers. For high availabilty, 
+do  you remeber where else on the chnanel we saw practice application of a consistent
+hashing algo. Right, designed distributed cache (read that first if you havent.). Another important concept is consistency. 
+
+
+every shard can talk to 2 other shards, no more than 3.  KNOWN AS Gossip protocol IN C*.
+
+
+When we defined nonfunctional requiements, we chose availability over consistency. We prefer to show stale data than no data at all. 
+Synchronous data replication is slow, we usually replicate asynchronoly.
+inconsistency is temporary, overtime, all writes will propogate to replicas, known as evential consistenc. 
+C* has eventual consistency, and has tunable consistency, the writes will reflect all nodes at some point. 
+
+
+How we store. 
+when designing data models for relational databases
+For relational, we define nouns, and use foreign keys to reference related data. 
+
+REPORT 
+
+INFO ABOUT VIDEO
+
+NUMBER OF VIEWS FOR LAST HOUR,  
+
+CHANNEL 
+
+need video_info table, video_stats table, channel_info table 
+
+generate report, run join query that gets data from all tables.
+
+Data is normalized, reduce data 
+
+
+https://www.youtube.com/watch?v=bUHFg8CZFws&t=3524s&ab_channel=SystemDesignInterview -> start watching from 25 mins 
+
+
+
+
+--------------------------------------------------------------------
+System Design Interview - Distributed Cache
+https://www.youtube.com/watch?v=iuqZvajTOyA&ab_channel=SystemDesignInterview
+
+Problem statement: 
+
+Client calls webservice which calls datastore. We want a cache in between. 
+Get data from distributed cache because we cannot store cache in memory since there is too much data. 
+
+Requirements: -> TELL INTERVIEWER ABOUT FUNCTIONAL AND NON FUNCTIONAL REQUIREMNTS
+Functional:
+We need to implement put and get
+Non functinal requirement: 
+
+Scalable (can deal with larger amounts of data ), Highly available (), High performance (fast puts and gets)
+
+NON FUNCTIONAL -> YOU CAN talk about CAP theorem.
+Interviwer is freidn -> needs to collect all data points. 
+evolve design with small steps from basic to adnvanced 
+Go from 
+LOCAL CACHE --> THEN GO TO DISTRIBUTED CACHE DURING INTERVIEW. 
+Local cache -> use hash table but also use LRU cache for eviction. (miultiple replacement policies. )
+LRU = map + doubly linked list. 
+
+System Design Interview - Distributed Cache
+DISCUSS Availability, CONSISTENCY, PARTITION TOLERANCE, AND Durability FOR DATABASES
+
+
+IMPELMENT LRU CACHE (LEETCODE) -> then split it based on consistent hashing of keys (SHARDED PER CACHE SERVER BASED ON KEY)
+
+Consistent hashing needs binary searching the cache server in the circle
+All cache clients should have same list of cache servers.
+Client stores list of servers in sorted order using treemap
+binary search used to identify server
+If server is unavailable client proceeds as though it was cache miss
+Cache client used in web serviec as library. 
+
+How do maintain list of cache servers that are available in conssitent hashing???
+First option
+store list of cache hosts in file, deploy to service host using continuous management tool pipelines like chef or puppet
+everytiem list changes, make code change, and deploy to every webservice 
+
+SECOND OPTION -> USE SHARED STORAGE -> AND SERVICE HOST POLLS FOR FILE
+
+HAVE FILE IN server. Can put the file in shared storage such as S3. All service hosts retrieve file from S3 storage
+Introudce daemon process THAT RUNS ON EACH SERVICE HOST, polls from S3 every minute to get file 
+drawback -> still have to maintain that file that contains all the configurations for the cache servers. 
+-> when cache host dies, have to change file, and same for adding cache servers 
+
+
+THIRD OPTION
+We should monitor CACHE server heath -> something bad happens -> then all web servers notified and stops sending requests to unavailable cache server 
+-> when new cache server is added -> all web service is notified -> send requests to it
+
+
+NEED NEW SERVICE -> CONFGIURATION SERVICE -> DISCOVER CACHE HOSTS AND MONITOR THEIR HEALTH -> AKA ZOOKEEPER (discover cache servers and monitor health)-> 
+CACHE SERVER SENDS HEARTBEATS TO CONFGIURATION SERVICE perioducy -> IF WE STOP GETTING HEARTBEATS IT BECOME DE-registered -> NEED THOSE HEARTBEATS!
+AND every cache client grabs list of cache servers from configiuration service. 
+
+ZOOKEEPER -> AUTOMATES LIST MAINTAINECE !!!
+
+connection between cache client and server is UDP/TCP. 
+
+Performance is there -> O(1) put and get and O(logn) to find server in consistent hash circle. 
+
+Scability is there -> can create more shards. Shards can become hot. Some shards process more than they appear. 
+Adding more cache servers may not be effective. -> we dont want to split all shards -> just the very hot one 
+We could fix it a diff way: 
+create more labels in the consistent hashing scheme. 20 labels per cache server in the consistent hash circle. 
+So 4 servers = 80 labels in the consistent hash circle. 
+
+High availability is not there at all yet! -> If some shard dies or becomes unavailbale due to network partition, all cache data for that shard is lost
+all requests to that shard will result in cache miss until keys are rehashed. 
+
+All cache data is lost, until keys are rehased. 
+
+Need to improve availability, and deal with hot shard problem 
+
+NEED DATA REPLICATION
+
+2 categories of data replication Protocols
+First category inclludes a set of probailitsitc protocols like gossip, epidemic broadcast trees, bimodal multicast.
+These protocols tend to favor eventual consistency
+
+The second category includes consensus protocols such as 2 or 3 phase commit, paxos, raft, chain replication,
+Test protocols tend to favor strong consistency 
+
+Lets keep things simple and use leader follower (also known as master slave) replication. 
+For each shard, designate master and bunch of read replicas, Read replicas try to be exact copy of master. Every time connection between master and replica breaks, 
+replica tries to reconnect to master.  Replicas try to automatically connect to master,
+each replica lives in diff data center so that cache data is still available when one data center is down. 
+All put calls go through master node, while get calls handled by master node and all replicas. 
+
+Get nodes handled by master and all replicas -> Can deal with hot shards by adding more read replicas
+
+Leader election: Rely on seperate component, configuration service (zookeeper), 
+or just if you want to avoid that, implement leader election in cache cluster.
+
+Configuration service responsible for monitoring of master and slaves, as well as failover. 
+If leader not working properly, promote follower to leader. Config service is source of authority for clients. 
+cache clients use config service to find all cache servers. Config service is distributed by nature, 
+and has odd # of nodes to achieve quorum easier, 
+nodes located on machines that fail independently so that configuration service remains aailale in case 
+for network partiitions. All nodes talk to each other with tcp protocol.
+CAN USE ZOOKEEPER OR REDIS SENTINEL. 
+
+Data replication deals with hot shards, and increased availability.
+But there are still points of failure. We do data replication replication async for better performance. 
+We do not want to wait until leader server replicates data to all the followers. 
+
+If leader server gets data and fails before replicated, then data is lost. This is acceptable behaviour for many real life use cases for cache. 
+
+Cache needs to be fast. Loses data in rare scenerios shouldnt be a big deal. Such cache failures are expected. thats just a cache miss doesnt matter.
+
+What topics will pop up during interview: 
+Distributed cache favours performance and availability over consistency
+
+THESE are the other things your should talk about in interview: 
+
+CONSISTENCY
+DATA expiration
+LOCAL AND REMOTE CACHE
+SECURTIY 
+MONITORING AND LOGGING
+Cache client
+Consistent hasshing
+
+
+Several things that lead to inconsistency: 
+Lead to inconsistency -> async data replication, -> dif read calls get diff results from diff replicas due to async
+another inconsistency -> clients may have different set of cache servers. Cache servers may go down and back up agian. 
+                         Clients write values that no other clients can read. Can fix this by introducing 
+                         synchronous replication and make sure all clients share a single view of cache servers list. 
+                         But this will increase latency and overall complexity of the system. Discuss the tradeoffs with your interviewer 
+
+                         LRU evicts data from cache when cache is full. But when cache isnt full, items stay a long time. Then when we do a get, 
+                         we get stale data. We can introduce TTL for cache entry by introducing metadata for each cache item. 
+                         Two common approaches to clean cache. 
+                         
+                         Passively expire an item, when some client tries to access it, check the ttl, and expire 
+                         if necessary. 
+
+                         Or we can actiely expire with maintence thread that runs at regular intervals and removes expired items. 
+                         As there may be billions of items in cache, we cant interate over all cache items. 
+                         usually some probabilistic algorithms are used, when several random items are tested with every run.. 
+
+Services that use distributed or remote cache often use local cache as well. If data isnt in local cache, then data is accessed from distributed cache. 
+
+To make life OF THE SERVICE TEAMS EASIER, WE CAN IMPLEMENT SUPPORT OF LOCAL CACHE IN CACHE CLIENT. HID COMPELLXITY BEHIND SINGLE COMPOENNT, 
+CACHE CLIENT INSTANT IS CREATED THEN WE CONSTRUCT LOCAL CACHE. HIDE COMPLEXITY BEHIND ONE CLIENT, THE CACHE CLIENT. 
+
+CACHES ARE OPTImized for pperofrmance, but not security. Caches are usually accessed by trusted clients, inside trusted environmentS. sHOULDNT 
+expose cache servERS directly to internet if it is not absolutely required. For this reason we should use firewall to restrict access to cache server ports, 
+and ensure only approved clients can access cache. 
+Clients may also encrypt data before storing in cache, and decrypt it on the way out. But we should expect performance implications. 
+
+Our cache has to be instrumented with metrics and logging, this is especially important, if we launch our distributed cache as a service, because
+so many service teams in the organization may use our cache, everytime those services experience perfomrance degradation, they will come to us as one
+of the potential sources of this degradation, and we should be able to answer their questison :
+
+What metrics should we emit:
+# OF FAULTS while calling cache, latency, HITS, MISSES, CPU AND MEM UTILIZATION, NETWORK I/O 
+
+logging capture details of every request to cache the basic infomration -> who, when, key, return status code (log entries should be small but usefl)
+
+Currently cache client has alot of responsibilties, pick shards, handle a remote call and any potential failures, emit metrics etc, 
+
+ideally Client software should be very simple, dumb if you want.  Can simplify cache client. 
+One idea is we can introduce proxy between cache client and cache servers, and responsible forpicking cache shard. Take a look twemproxy project created by twitter. 
+
+Another idea, is to make cache servers responbile for picking a shard. 
+Client sends request to random cache server, and cache server applies consistent hashing or some other partitioning algorithm, and redirects request to shard 
+that stores data. This idea is utilized by redis cluster. 
+Consistent hashing algorithm is great, simple, and effective, but it has 2 major falls. 
+So called domino effect, and the fact that cache servers do not split circle evenly. Domino effect may appear when cache server dies, and all of its load 
+is transferred to next server, this transfer might overload next server and then that server would fail, causing a chain reaction of failures, and to understand 
+second problem, remember how we placed cache servers on circle, based on hashing their ip address. Some servers might be close together, while others are 
+far apart, dealing with uneven distribution of keys. 
+
+
+Several modifcations for consisten hash algorithm :
+Simple idea: Add each server on the circle multiple times, you can also read about jump hash algorithm a paper published by google on 2014,
+or proportional hasing (algorithm used by yahoo video platform)
+
+Quick recap of what we discussed:
+Single host, and have local cache, 
+
+Local cache has not enuff mem -> so create a standolne service, 
+Then use consistent caching for servers, and have cache client that routes requests for each key to specific shards that store data for key. 
+
+Can stop right here, memcached, is built upon just those principles 
+
+We went further and improved scalability and failover support, with configuration service. 
+
+
+---------------------------------------------------------------------------------------------------------------------------------
+
+
+https://leetcode.com/discuss/study-guide/901324/My-System-Design-Interview-Checklist-A-Gateway-to-FAANGs

+Usually, the System Design interviews are lengthy and cover a lot of complex components. This makes it very easy to get 
+lost in small things and miss out on the big picture. Therefore, it becomes very important to structure your interview in a 
+way such that you can easily convey the whole picture without wasting time on anything which does not add value.
+
+Below is the structure that I follow, and you could try the same.
+
+Drive the interview
+Make sure you are the one driving the interview and not your interviewer. This does not mean that you do not let them speak, but rather, you should be the one doing most of the talking, proactively calling out issues in your design before the interviewer points it out, handle the edge cases that the interviewer might poke you on etc.
+
+FRs and NFRs
+Clearly call out the Functional and Non-Functional requirements.
+
+The intent is that the requirements should be big enough that makes the problem challenging and also finite enough 
+that you can build a system that fulfills those requirements within the stipulated time. From the Non Functional side, try 
+to make a system that works at a very large scale. What's the fun in designing a system which works at a low scale?
+
+Before finalizing the FRs and the NFRs, get them reviewed with your interviewer to make sure they do not want to 
+    add/ remove something. At times, interviewers do have some particular use cases in mind that they want to go over.
+
+Capacity Estimation
+Check with your interviewer if they want to get into this. A lot of people prefer to skip the calculations and focus more on the design, assuming a large enough approximate number for the number of cores required or the amount of disk required etc.
+
+Plan
+Based on the FRs and NFRs, come up with these things:
+
+User Interaction Points.
+Latency/ Availability/ Consistency requirements at each of the user interaction points.
+A quick analysis estimating if it's a read-heavy interaction or a write-heavy interaction.
+Based on the above three, come up with what all services you'll need and what kind of databases you can use to store the data that each of these services owns.
+HLD
+Come up with a high level component diagram, that covers the following:
+
+What all services are present? Make sure you divide the flow into multiple functional components and see if a microservices 
+    based approach makes sense or not. Usually, using a microservices based approach is a good idea in SD interviews.
+How do the services interact with each other and what kind of protocols are used for inter service communication like Async/ Sync - Rest, RPC etc?
+How would the users interact with the whole system and what all services are user facing. Do you need a Cache to reduce latencies?
+Which service uses what Database and why? You can refer to this article that can help you choose the right database based on your use case
+See if you need caching anywhere and if you do, then what shall be the eviction policy, do you need an expiry for the keys, should it be a write through cache etc?
+Basis all this analysis, draw out a High Level Diagram of your whole system.
+Must Haves
+Some key things your high level diagram should have are:
+
+Load Balancers
+Services
+Databases and Caches
+User interaction points
+Any other tools like a Message Queue, CDN, etc.
+Walkthrough the design
+Once you have the whole diagram ready, go over the whole design, one use case at a time and explain your design to your interviewer at a very high level.
+Talk about why you have chosen a particular database here and why you have used a particular mode of communication like Sync/ Async etc. You can also get into an RPC vs HTTP kind of a conversation if you made a particular design choice. You should go over what kind of data replication strategy is being used in your databases, for example, would you use a Master-Slave or a Multi Master setup etc.
+
+If this sounds intimidating, you can check out how I usually do a design walkthrough in this video.
+
+
+
+CAUTION: Do not go into the details like APIs, DB Schema etc right away unless the interviewer asks for it. Most people get lost in designing the APIs for just one system at this point and run out of time later on.
+
+Brownie Points: Most interviews do not have FRs and NFRs around analytics, but if your design covers that or leaves good enough scope for analytics, that elevates your solution a lot. Try to add that. For example, you can look at this.
+
+This video also covers Analytics in great depth.
+
+
+
+Implementation
+Once you explain the whole flow to your interviewer, ask them which component they want to discuss in detail.
+Usually, people do not want to go over the entire system in detail. Let them decide which is that one component 
+that they want to dig into and then you can go over the implementation details of that particular system.
+Things you should cover here are:
+
+APIs- Call out the APIs that this system exposes. Make sure you are using the best practices here. For example instead of a GET API with URL like GET /user/getUserbyUserId, it’s better to use: GET /user/{id}
+API Protocols - You can cover what protocols are you exposing the APIs on. Most people choose REST APIs, but you can decide to use something more efficient like Thrift, Protobuf etc based on your use cases.
+Events - You can call out which events this particular service listens to, who produces that event, what payload comes in, what processing happens on that event etc.
+DB Schema - Go over the DB Schema here. You can also get into SQL vs NoSQL debate or why have you chosen a particular database, if you did not go over the same earlier while talking about the high level design.
+If it's a SQL, do talk about what indices you'll have and how are you optimising your queries. In case of NoSQL, make sure you go over the consistency guarantees that the DB provides, can it cause any issues and the kind of queries you'll run on that DB. Clearly call out the keys for key-value stores or the partition keys for a columnar store etc.
+Handle Murphy's law
+This is something that most people skip but this is one of the most important things that you must cover which talks about how resilient your system is. In the real world, things break, and when they do, you need to make sure you are in full control of your system.
+
+Talk about how you monitor the system. What kind of alerting mechanism do you have in place? What are your KPIs (Key Performance Indicators) and how do you track them? What happens when things break, when your service crashes, your DB's master node goes down or even when one of your datacentres goes down?
+
+Again, if you haven't done this before, see how I have been doing it, towards the later half of this video.
+
+
+
+This is my checklist, that I usually follow when I try to design any system, be it in an interview or in the real world.
+
+Thoughts/Suggestions?


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

Luckily, there’s a paper that solves this. In 1997, the paper “Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web” 
was released. This paper described the approach used by Akamai in their distributed content delivery network.

It took until 2007 for the ideas to seep into the popular consciousness. That year saw two works published:
last.fm’s Ketama memcached client.

Dynamo: Amazon’s Highly Available Key-value Store
These cemented consistent hashing’s place as a standard scaling technique. It’s now used by Cassandra, 
Riak, and basically every other distributed system that needs to distribute load over servers.
This algorithm is the popular ring-based consistent hashing. You may have seen a “points-on-the-circle” diagram. 
When you do an image search for “consistent hashing”, this is what you get:
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

    Federation (or functional partitioning) splits up databases by function. For example, instead of a single, 
    monolithic database, you could have three databases: forums, users, and products, resulting in less read 
    and write traffic to each database and therefore less replication lag. Smaller databases result in more data 
    that can fit in memory, which in turn results in more cache hits due to improved cache locality. With no 
    single central master serializing writes you can write in parallel, increasing throughput.

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

#######
####################


🔹
Serializalble: This is the highest isolation level. Concurrent transactions are guaranteed to be executed in sequence.
🔹
Repeatable Read: Data read during the transaction stays the same as the transaction starts.
🔹
Read Committed: Data modification can only be read after the transaction is committed.
4
🔹
Read Uncommitted: The data modification can be read by other transactions before a transaction is committed.

The isolation is guaranteed by MVCC (Multi-Version Consistency Control) and locks.
The diagram below takes Repeatable Read as an example to demonstrate how MVCC works:
There are two hidden columns for each row: transaction_id and roll_pointer. When transaction A starts, a new Read View with transaction_id=201 is created. Shortly afterward, transaction B starts, and a new Read View with transaction_id=202 is created.
Now transaction A modifies the balance to 200, a new row of the log is created, and the roll_pointer points to the old row. Before transaction A commits, transaction B reads the balance data. Transaction B finds that transaction_id 201 is not committed, it reads the next committed record(transaction_id=200).
Even when transaction A commits, transaction B still reads data based on the Read View created when transaction B starts. So transaction B always reads the data with balance=100.


FAN OUT PATTERN:

The textbook example of the fan-out design pattern is Twitter's news feed (i.e. timeline) generation service. 
The naive approach would consist of writing tweets to a database. Then, whenever the followers open the application, the client would request the tweets from the database. As you could image, if you have millions of users and each user follows thousands of people, this would place a signicant load on the database.
A better approach consists of pre-computing the user's timeline and storing it in a Redis cluster. In other words, when a user posts a tweet, the tweet is inserted into the timeline of each of their followers. For example, if a user has 1000 followers, then their tweet would result in 1000 writes.
The process of writing the same data to multiple destinations simultaneously is called fan-out.
It's worth noting that in the case of celebrities with millions of followers, we don't fan-out their tweets. Rather, the users that follow said celebrity will request the data only when they open the application.
In one of my previous roles, we were streaming data in JSON format to our data warehouse. The amount of data was so large that we needed to create indices using separate tables that were partitioned based on different dimensions. For example, we'd have a table containing two columns, one for the geographical location and another for the record id such that the analysts could easily determine what records could be found in that region.
Initially, we had accomplished the latter using multiple independent streams. However, we were throttling the message broker since we had multiple consumers attempting to read the same data.
When using Spark Structured Streams, we can write to multiple destinations using the `foreachBatch` method. After making the switch, we noticed a significant performance boost. When writing to multiple Delta tables, if we specify a txnVersion and txnAppId, Databricks provides guarantees that these streaming writes will be idempotent. This is especially useful when data for multiple tables will be contained within a single record since we'd want to rollback in the event of a failure. It's worth noting that at the time of this writing, I believe this is not yet supported for non Delta tables.
Here's an example of how we'd write to multiple tables simultaneously using Spark:
def write_twice(microBatchDF, batchId)
	appId = 'write_twice'
	microBatchDF.select(
		'id',
		'name',
		F.current_timestamp().alias('processed_time')
	) \
	.write \
	.option('txnVersion', batchId) \
	.option('txnAppId', appId) \
	.mode('append') \
	.saveAsTable('silver_name')


	microBatchDF.select(
		'id',
		'value',
		F.current_timestamp().alias('processed_time')
	) \
	.write \
	.option('txnVersion', batchId) \
	.option('txnAppId', appId) \
	.mode('append') \
	.saveAsTable('silver_value')


def split_stream():
	query = (spark.readStream
			.table('bronze')
			.writeStream
			.foreachBatch(write_twice)
			.option('checkpointLocation', checkpoint_location)
			.trigger(availableNow=True)
			.start())
	query.awaitTermination():
Can you think of any other applications of the fan-out pattern? Leave a comment below.


#######
#########

CONSISTENCY:



Strong Consistency See all previous writes.
Eventual Consistency See subset of previous writes.
Consistent Prefix See initial sequence of writes.
Bounded Staleness See all “old” writes.
Monotonic Reads See increasing subset of writes.
Read My Writes See all writes performed by reader.
Table 1. Six Consistency Guarantees

Guarantee Consistency Performance Availability
Strong Consistency excellent poor poor
Eventual Consistency poor excellent excellent
Consistent Prefix okay good excellent
Bounded Staleness good okay poor
Monotonic Reads okay good good
Read My Writes okay okay okay


Strong consistency is particularly easy to understand. It guarantees that a read operation returns the value that
was last written for a given object. If write operations can modify or extend portions of a data object, such as
appending data to a log, then the read returns the result of applying all writes to that object. In other words, a
read observes the effects of all previously completed writes.

Eventual consistency is the weakest of the guarantees, meaning that it allows the greatest set of possible return
values. For whole-object writes, an eventually consistent read can return any value for a data object that was
written in the past. More generally, such a read can return results from a replica that has received an arbitrary
subset of the writes to the data object being read.

By requesting a consistent prefix, a reader is guaranteed to observe an ordered sequence of writes starting with
the first write to a data object. For example, the read may be answered by a replica that receives writes in order
from a master replica but has not yet received an unbounded number of recent writes. In other words, the
reader sees a version of the data store that existed at the master at some time in the past. This is similar to the
“snapshot isolation” consistency offered by many database management systems.

Bounded staleness ensures that read results are not too out-of-date. Typically, staleness is defined by a time
period T, say 5 minutes. The storage system guarantees that a read operation will return any values written
more than T minutes ago or more recently written values. Alternative, some systems have defined staleness in
terms of the number of missing writes or even the amount of inaccuracy in a data value. I find that timebounded staleness is the 
most natural concept for application developers.

Monotonic Reads is a property that applies to a sequence of read operations that are performed by a given
storage system client. As such, it is often called a “session guarantee.” With monotonic reads, a client can read
arbitrarily stale data, as with eventual consistency, but is guaranteed to observe a data store that is increasingly
up-to-date over time. In particular, if the client issues a read operation and then later issues another read to the
same object(s), the second read will return the same value(s) or the results of later writes.
4

Read My Writes is a property that also applies to a sequence of operations performed by a single client. It
guarantees that the effects of all writes that were performed by the client are visible to the client’s subsequent
reads. If a client writes a new value for a data object and then reads this object, the read will return the value
that was last written by the client (or some other value that was later written by a different client). (Note: In
other papers, this has been called “Read Your Writes,” but I have chosen to rename it to more accurately
describe the guarantee from the client’s viewpoint.)

These last four read guarantees are all a form of eventual consistency but stronger than the eventual
consistency model that is typically provided in systems like Amazon. None of these four guarantees is stronger
than any of the others, meaning that each might result in a read operation returning a different value. In some
cases, as will be shown later, applications may want to request multiple of these guarantees. For example, a
client could request monotonic reads and read my writes so that it observes a data store that is consistent with
its own actions.


#######

WIKIMEDIA ARCHITECTURE:

Lessons Learned
Focus on architecture, not so much on operations or nontechnical stuff.

Sometimes caching costs more than recalculating or looking up at the
data source...profiling!

Avoid expensive algorithms, database queries, etc.

Cache every result that is expensive and has temporal locality of reference.

Focus on the hot spots in the code (profiling!).

Scale by separating:
- Read and write operations (master/slave)
- Expensive operations from cheap and more frequent operations (query groups)
- Big, popular wikis from smaller wikis

Improve caching: temporal and spatial locality of reference and reduces the data set size per server

Text is compressed and only revisions between articles are stored.

Simple seeming library calls like using stat to check for a file's existence can take too long when loaded.

Disk seek I/O limited, the more disk spindles, the better!

Scale-out using commodity hardware doesn't require using cheap hardware. Wikipedia's database servers these days are 16GB dual or quad core boxes with 6 15,000 RPM SCSI drives in a RAID 0 setup. That happens to be the sweet spot for the working set and load balancing setup they have. They would use smaller/cheaper systems if it made sense, but 16GB is right for the working set size and that drives the rest of the spec to match the demands of a system with that much RAM. Similarly the web servers are currently 8 core boxes because that happens to work well for load balancing and gives good PHP throughput with relatively easy load balancing.

It is a lot of work to scale out, more if you didn't design it in originally. Wikipedia's MediaWiki was originally written for a single master database server. Then slave support was added. Then partitioning by language/project was added. The designs from that time have stood the test well, though with much more refining to address new bottlenecks.

Anyone who wants to design their database architecture so that it'll allow them to inexpensively grow from one box rank nothing to the top ten or hundred sites on the net should start out by designing it to handle slightly out of date data from replication slaves, know how to load balance to slaves for all read queries and if at all possible to design it so that chunks of data (batches of users, accounts, whatever) can go on different servers. You can do this from day one using virtualisation, proving the architecture when you're small. It's a LOT easier than doing it while load is doubling every few months!


----------------------------------------------------------------

### **Basic Algorithm of 2PC**

### **`prepare` phase**

The coordinator sends a `prepare` message to all cohorts and waits until it has received a reply from all cohorts.

### **`commit` phase**

If the coordinator received an agreement message from all cohorts during the `prepare` phase, the coordinator sends a `commit` message to all the cohorts.

If any cohort votes `No` during the `prepare` phase (or the coordinator’s timeout expires), the coordinator sends a `rollback` message to all the cohorts.

### **Disadvantages of 2PC**

The greatest disadvantage of the two-phase commit protocol is that it is a blocking protocol. If the coordinator fails permanently, some cohorts will never resolve their transactions: after a cohort has sent an agreement message to the coordinator, it will block until a `commit` or `rollback` is received.

For example, consider a transaction involving a coordinator `A` and the cohort `C1`. If `C1` receives a `prepare` message and responds to `A`, then `A` fails before sending `C1` either a `commit` or `rollback` message, then `C1` will block forever.

### **2PC Practice in TiKV**

In TiKV we adopt the [Percolator transaction model](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36726.pdf) which is a variant of two phase commit. To address the disadvantage of coordinator failures, percolator doesn’t use any node as coordinator, instead it uses one of the keys involved in each transaction as a coordinator. We call the coordinating key the primary key, and the other keys secondary keys. Since each key has multiple replicas, and data is kept consistent between these replicas by using a consensus protocol (Raft in TiKV), one node’s failure doesn’t affect the accessibility of data. So Percolator can tolerate node fails permanently.

## **Three-Phase Commit**

Unlike the two-phase commit protocol (2PC), 3PC is non-blocking. Specifically, 3PC places an upper bound on the amount of time required before a transaction either commits or aborts. This property ensures that if a given transaction is attempting to commit via 3PC and holds some resource locks, it will release the locks after the timeout.

### **1st phase**

The coordinator receives a transaction request. If there is a failure at this point, the coordinator aborts the transaction. Otherwise, the coordinator sends a `canCommit?` message to the cohorts and moves to the waiting state.

### **2nd phase**

If there is a failure, timeout, or if the coordinator receives a `No` message in the waiting state, the coordinator aborts the transaction and sends an `abort` message to all cohorts. Otherwise the coordinator will receive `Yes` messages from all cohorts within the time window, so it sends `preCommit` messages to all cohorts and moves to the prepared state.

### **3rd phase**

If the coordinator succeeds in the prepared state, it will move to the commit state. However if the coordinator times out while waiting for an acknowledgement from a cohort, it will abort the transaction. In the case where an acknowledgement is received from the majority of cohorts, the coordinator moves to the commit state as well.

A two-phase commit protocol cannot dependably recover from a failure of both the coordinator and a cohort member during the Commit phase. If only the coordinator had failed, and no cohort members had received a commit message, it could safely be inferred that no commit had happened. If, however, both the coordinator and a cohort member failed, it is possible that the failed cohort member was the first to be notified, and had actually done the commit. Even if a new coordinator is selected, it cannot confidently proceed with the operation until it has received an agreement from all cohort members, and hence must block until all cohort members respond.

The three-phase commit protocol eliminates this problem by introducing the `Prepared-to-commit` state. If the coordinator fails before sending `preCommit` messages, the cohort will unanimously agree that the operation was aborted. The coordinator will not send out a `doCommit` message until all cohort members have acknowledged that they are `Prepared-to-commit`. This eliminates the possibility that any cohort member actually completed the transaction before all cohort members were aware of the decision to do so (an ambiguity that necessitated indefinite blocking in the two-phase commit protocol).

### **Disadvantages of 3PC**

The main disadvantage to this algorithm is that it cannot recover in the event the network is segmented in any manner. The original 3PC algorithm assumes a fail-stop model, where processes fail by crashing and crashes can be accurately detected, and does not work with network partitions or asynchronous communication.

The protocol requires at least three round trips to complete. This potentially causes a long latency in order to complete each transaction.

## **Paxos Commit**

The Paxos Commit algorithm runs a Paxos consensus algorithm on the commit/abort decision of each participant to achieve a transaction commit protocol that uses 2F + 1 coordinators and makes progress if at least F + 1 of them are working properly. Paxos Commit has the same stable-storage write delay, and can be implemented to have the same message delay in the fault-free case, as Two-Phase Commit, but it uses more messages. The classic Two-Phase Commit algorithm is obtained as the special F = 0 case of the Paxos Commit algorithm.

In the Two-Phase Commit protocol, the coordinator decides whether to abort or commit, records that decision in stable storage, and informs the cohorts of its decision. We could make that fault-tolerant by simply using a consensus algorithm to choose the committed/aborted decision, letting the cohorts be the client that proposes the consensus value. This approach was apparently first proposed by Mohan, Strong, and Finkelstein, who used a synchronous consensus protocol. However, in the normal case, the leader must learn that each cohort has prepared before it can try to get the value committed chosen. Having the cohorts tell the leader that they have prepared requires at least one message delay.
