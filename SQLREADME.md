# REVIEW THIS BEFORE YOU REVIEW THE BELOW GUIDE FOR 100 SQL QUERIES :

https://gvwilson.github.io/sql-tutorial/core/

https://hakibenita.com/sql-dos-and-donts


SELECT id FROM table WHERE EXTRACT(year FROM creation_date) = 2010


DATA MODELLING :

1. What is data modeling?
2. What are the objectives of data modeling?
3. What are the three levels of data modeling?
4. What is conceptual data model?
5. What is logical data model?
6. What is physical data model?
7. What is entity relationship diagram (ERD)?
8. What are the components of ERD?
9. What is normalization? What are the normal forms?
10. What is denormalization? When is it used?

SQL :

11. What is SQL?
12. What are the types of SQL commands?
13. What is the difference between WHERE and HAVING clause?
14. What is the difference between UNION and UNION ALL?
15. What is the difference between DELETE and TRUNCATE statements?
16. What is the difference between primary key and unique constraint?
17. What is a view? What are the benefits of views?
18. How to create a view?
19. What is a stored procedure? What are the benefits of stored procedures?
20. How to create a stored procedure?

TRANSACTION :

21. What is a transaction in DBMS?
22. What are the ACID properties of transactions?
23. What is a commit operation?
24. What is a rollback operation?
25. What is a deadlock in transactions?
26. What is a transaction log?
27. What is a savepoint in transactions?
28. What is concurrency control?
29. What are the types of transaction isolation levels?
30. What is two-phase locking (2PL) protocol?

KEYS AND CONSTRAINTS :

31. What is a key in DBMS?
32. What is a primary key?
33. What is a foreign key?
34. What is a composite key?
35. What is a candidate key?
36. What is a unique key?
37. What is a constraint in DBMS?
38. What is a NOT NULL constraint?
39. What is a CHECK constraint?
40. What is a referential integrity constraint?

Database Normalization:

41. What is database normalization?
42. What are the main objectives of normalization?
43. What is the First Normal Form (1NF)?
44. What is the Second Normal Form (2NF)?
45. What is the Third Normal Form (3NF)?
46. What is the Boyce-Codd Normal Form (BCNF)?
47. What is the Fourth Normal Form (4NF)?
48. What is the Fifth Normal Form (5NF)?
49. What is denormalization, and when is it used?
50. What are the advantages and disadvantages of normalization?

Operating System

Compiler, Loader, Linker
Process
Threads
Process Scheduling
Paging
Segmentation
Mutex/Semaphores
DBMS

Normalisation
Locking
Concurrency Control
File vs DBMS
SQL vs NoSQL
Indexing
ACID Properties
Computer Networks

OSI Model
Subnetting/ Supernetting
How Internet Works
Routing Algorithms
IPv4 vs IPv6
http vs https
OOPS

Class
Object
Abstraction
Encapsulation
Inheritance
Polymorphism
Linux

Inode
File Structure
Priority in Linux
fork()/ pipe()
Multithreading
How OS works?
Basic commands
System Design

CAP theorem
Scaling
Load Balancer
Distributed System
Caching and its types
Sharding
API (REST vs SOAP)



# The Best Medium-Hard Data Analyst SQL Interview Questions  

By Zachary Thomas ([zthomas.nc@gmail.com](mailto:zthomas.nc@gmail.com), [Twitter](https://twitter.com/zach_i_thomas), [LinkedIn](https://www.linkedin.com/in/thomaszi/)) 

**Tip: **See the Table of Contents (document outline) by hovering over the vertical line on the right side of the page 

**Update:** Thanks everyone for the support and feedback! See this discussion on this post on [Hacker News](https://news.ycombinator.com/item?id=23053981), [Linkedin](https://www.linkedin.com/posts/thomaszi_the-best-medium-hard-data-analyst-sql-interview-activity-6662828341382516736-L5If), Eric Weber’s [Linkedin post](https://www.linkedin.com/posts/eric-weber-060397b7_datascience-analytics-sql-activity-6663082042990952449-wuOK) 

## Background & Motivation

> The first 70% of SQL is pretty straightforward but the remaining 30% can be pretty tricky.


Between the fall of 2015 and the summer of 2019 I interviewed for data analyst and data scientists positions four separate times, getting to onsite interviews at over a dozen companies. After an interview in 2017 went poorly — mostly due to me floundering at the more difficult SQL questions they asked me — I started putting together a study guide of medium and hard SQL questions to better prepare and found it particularly useful during my 2019 interview cycle. Over the past year I have shared that guide with a couple of friends, and with the extra time on my hands due to the coronavirus pandemic, I have polished it up into this doc. 

There are plenty of great beginner SQL guides out there. My favorites are Codecademy’s [interactive SQL courses](https://www.codecademy.com/learn/learn-sql) and Zi Chong Kao’s [Select Star SQL](https://selectstarsql.com/). However, like I told a friend, while the first 70% of SQL is pretty straightforward, the remaining 30% can be pretty tricky. Data analyst and data scientist interview questions at technology companies often pull from that 30%.  

Strangely, I have never really found a comprehensive source online for those medium-hard SQL questions, which is why I put together this guide. 

Working through this guide should improve your performance on data analyst interviews. It should also make you better at your current and future job positions. Personally, I find some of the SQL patterns found in this doc useful for ETLs powering reporting tools featuring trends over time. 

To be clear, data analyst and data scientist interviews consist of more than SQL questions. Other common topics include explaining past projects, A/B testing (I like [Udacity’s course](https://www.udacity.com/course/ab-testing--ud257) on the subject), metric development and open-ended analytical problems. This [Quora answer](https://qr.ae/pNrdGV) has Facebook’s product analyst interview guide circa 2017, which discusses this topic in more depth. That said, if improving your SQL skills can make your interviews less stressful than they already are, it could very well be worth your time. 

In the future, I may transition this doc to a website like [Select Star SQL](https://selectstarsql.com/) with an embedded SQL editor so that readers can write SQL statements to questions and get real-time feedback on their code. Another option could be adding these questions as problems on Leetcode. For the time being though I just wanted to publish this doc so that people could find it useful now.  

**I would love to get your feedback on this doc. Please drop a note if you find this useful, have improvements/corrections, or encounter other good resources for medium/hard difficulty SQL questions. **

## Assumptions & How to use this guide 

**Assumptions about SQL proficiency: **This guide assumes you have a working knowledge of SQL. You probably use it frequently at work already but want to sharpen your skills on topics like self-joins and window functions. 

**How to use this guide:** Since interviews usually utilize a whiteboard or a virtual (non-compiling) notepad, my recommendation is to get out a pencil and paper and write out your solutions to each part of the problem, and once complete compare your answers to the answer key. Or, complete these with a friend who can act as the interviewer!

* Small SQL syntax errors aren’t a big deal during whiteboard/notepad interviews. However, they can be distracting to the interviewer, so ideally practice reducing these so your logic shines through in the interview. 
* The answers I provide may not be the only way to successfully solve the question. Feel free to message with additional solutions and I can add them to this guide! 

## Tips on solving difficult SQL interview questions 

This advice mirrors typical code interview advice ... 

1. Listen carefully to problem description, repeat back the crux of the problem to the interviewer
2. Spell out an edge case to demonstrate you actually understand problem (i.e. a row that *wouldn’t* be included in the output of the SQL you are about to sketch out) 
3. (If the problem involves a self-join) For your own benefit sketch out what the self-join will look like — this will typically be at least three columns: a column of interest from the main table, the column to join from the main table, and the column to join from the secondary table 
    1. Or, as you get more used to self-join problems, you can explain this step verbally 
4. Start writing SQL — err towards writing SQL versus trying to perfectly understand the problem. Verbalize your assumptions as you go so your interviewer can correct you if you go astray. 

## Acknowledgments and Additional Resources 

Some of the problems listed here are adapted from old Periscope blog posts (mostly written around 2014 by [Sean Cook](https://www.linkedin.com/in/seangcook/), although his authorship seems to have been removed from the posts following SiSense's [merger with](https://www.sisense.com/blog/sisense-and-periscope-data-merge-2/) Periscope) or discussions from Stack Overflow; I've noted them at the start of questions as appropriate. 

[Select Star SQL](https://selectstarsql.com/) has good[challenge questions](https://selectstarsql.com/questions.html#challenge_questions) that are complementary to the questions in this doc. 

Please note that these questions are not literal copies of SQL interview questions I have encountered while interviewing nor were they interview questions used at a company I have worked at or work at. 
* * *

# Self-Join Practice Problems 

## #1: MoM Percent Change 

**Context:** Oftentimes it's useful to know how much a key metric, such as monthly active users, changes between months. Say we have a table `logins` in the form: 

```
| user_id | date       |
|---------|------------|
| 1       | 2018-07-01 |
| 234     | 2018-07-02 |
| 3       | 2018-07-02 |
| 1       | 2018-07-02 |
| ...     | ...        |
| 234     | 2018-10-04 |
```

**Task**: Find the month-over-month percentage change for monthly active users (MAU). 
* * *
***Solution:***

*(This solution, like other solution code blocks you will see in this doc, contains comments about SQL syntax that may differ between flavors of SQL or other comments about the solutions as listed) *

```
WITH mau AS 
(
  SELECT 
   /* 
    * Typically, interviewers allow you to write psuedocode for date functions 
    * i.e. will NOT be checking if you have memorized date functions. 
    * Just explain what your function does as you whiteboard 
    *
    * DATE_TRUNC() is available in Postgres, but other SQL date functions or 
    * combinations of date functions can give you a identical results   
    * See https://www.postgresql.org/docs/9.0/functions-datetime.html#FUNCTIONS-DATETIME-TRUNC
    */ 
    DATE_TRUNC('month', date) month_timestamp,
    COUNT(DISTINCT user_id) mau
  FROM 
    logins 
  GROUP BY 
    DATE_TRUNC('month', date)
  )
 
 SELECT 
    /*
    * You don't literally need to include the previous month in this SELECT statement. 
    * 
    * However, as mentioned in the "Tips" section of this guide, it can be helpful 
    * to at least sketch out self-joins to avoid getting confused which table 
    * represents the prior month vs current month, etc. 
    */ 
    a.month_timestamp previous_month, 
    a.mau previous_mau, 
    b.month_timestamp current_month, 
    b.mau current_mau, 
    ROUND(100.0*(b.mau - a.mau)/a.mau,2) AS percent_change 
 FROM
    mau a 
 JOIN 
    /*
    * Could also have done `ON b.month_timestamp = a.month_timestamp + interval '1 month'` 
    */
    mau b ON a.month_timestamp = b.month_timestamp - interval '1 month' 
  
```



## #2: Tree Structure Labeling   

**Context:** Say you have a table `tree` with a column of nodes and a column corresponding parent nodes 

```
node   parent
1       2
2       5
3       5
4       3
5       NULL 
```

**Task:** Write SQL such that we label each node as a “leaf”, “inner” or “Root” node, such that for the nodes above we get: 

```
node    label  
1       Leaf
2       Inner
3       Inner
4       Leaf
5       Root
```

A solution which works for the above example will receive full credit, although you can receive extra credit for providing a solution that is generalizable to a tree of any depth (not just depth = 2, as is the case in the example above). 

(Side note: [this link](http://ceadserv1.nku.edu/longa//classes/mat385_resources/docs/trees.html) has more details on Tree data structure terminology. Not needed to solve the problem though!)
* * *
***Solution:***

**Note: **This solution works for the example above with tree depth = 2, but is not generalizable beyond that. 

```
WITH join_table AS 
(
SELECT 
    a.node a_node,
    a.parent a_parent,
    b.node b_node, 
    b.parent b_parent
 FROM
    tree a 
 LEFT JOIN 
    tree b ON a.parent = b.node 
 )
 
 SELECT 
    a_node node, 
    CASE 
        WHEN b_node IS NULL and b_parent IS NULL THEN 'Root'
        WHEN b_node IS NOT NULL and b_parent IS NOT NULL THEN 'Leaf'        
        ELSE 'Inner' 
    END AS label 
 FROM 
    join_table 
        
```

An alternate solution, that is generalizable to any tree depth: 

**Acknowledgement:** this more generalizable solution was contributed by Fabian Hofmann on 5/2/20. Thank, FH! 

```
WITH join_table AS
(
    SELECT 
        cur.node, 
        cur.parent, 
        COUNT(next.node) AS num_children
    FROM 
        tree cur
    LEFT JOIN 
        tree next ON (next.parent = cur.node)
    GROUP BY 
        cur.node, 
        cur.parent
)

SELECT
    node,
    CASE
        WHEN parent IS NULL THEN "Root"
        WHEN num_children = 0 THEN "Leaf"
        ELSE "Inner"
    END AS label
FROM 
    join_table 
```

An alternate solution, without explicit joins: 

**Acknowledgement:** William Chargin on 5/2/20 noted that `WHERE parent IS NOT NULL`  is needed to make this solution return `Leaf` instead of `NULL`. Thanks, WC! 

```
SELECT 
    node,
    CASE 
        WHEN parent IS NULL THEN 'Root'
        WHEN node NOT IN 
            (SELECT parent FROM tree WHERE parent IS NOT NULL) THEN 'Leaf'
        WHEN node IN (SELECT parent FROM tree) AND parent IS NOT NULL THEN 'Inner'
    END AS label 
 from 
    tree
```



## #3: Retained Users Per Month (multi-part)

**Acknowledgement: **this problem is adapted from SiSense’s [“Using Self Joins to Calculate Your Retention, Churn, and Reactivation Metrics”](https://www.sisense.com/blog/use-self-joins-to-calculate-your-retention-churn-and-reactivation-metrics/) blog post

### Part 1: 

**Context:** Say we have login data in the table `logins`: 

```
| user_id | date       |
|---------|------------|
| 1       | 2018-07-01 |
| 234     | 2018-07-02 |
| 3       | 2018-07-02 |
| 1       | 2018-07-02 |
| ...     | ...        |
| 234     | 2018-10-04 |
```

**Task:** Write a query that gets the number of retained users per month. In this case, retention for a 
given month is defined as the number of users who logged in that month who also logged in the immediately previous month. 
* * *
***Solution:***

```
SELECT 
    DATE_TRUNC('month', a.date) month_timestamp, 
    COUNT(DISTINCT a.user_id) retained_users 
 FROM 
    logins a 
 JOIN 
    logins b ON a.user_id = b.user_id 
        AND DATE_TRUNC('month', a.date) = DATE_TRUNC('month', b.date) + 
                                             interval '1 month'
 GROUP BY 
    date_trunc('month', a.date)
```

**Acknowledgement: **Tom Moertel pointed out de-duping user-login pairs before the self-join would make the solution more efficient and contributed the alternate solution below. Thanks, TM! 

**Note: **De-duping `logins` would also make the given solutions to Parts 2 and 3 of this problem more efficient as well.

Alternate solution: 

```
WITH DistinctMonthlyUsers AS (
  /*
  * For each month, compute the *set* of users having logins.
  */
    SELECT DISTINCT
      DATE_TRUNC('MONTH', date) AS month_timestamp,
      user_id
    FROM logins
  )

SELECT
  CurrentMonth.month_timestamp month_timestamp,
  COUNT(PriorMonth.user_id) AS retained_user_count
FROM 
    DistinctMonthlyUsers AS CurrentMonth
LEFT JOIN 
    DistinctMonthlyUsers AS PriorMonth
  ON
    CurrentMonth.month_timestamp = PriorMonth.month_timestamp + INTERVAL '1 MONTH'
    AND 
    CurrentMonth.user_id = PriorMonth.user_id
```

### Part 2: 

**Task:** Now we’ll take retention and turn it on its head: Write a query to find many users 
last month *did not* come back this month. i.e. the number of churned users.  
* * *
***Solution:***

```
SELECT 
    DATE_TRUNC('month', a.date) month_timestamp, 
    COUNT(DISTINCT b.user_id) churned_users 
FROM 
    logins a 
FULL OUTER JOIN 
    logins b ON a.user_id = b.user_id 
        AND DATE_TRUNC('month', a.date) = DATE_TRUNC('month', b.date) + 
                                         interval '1 month'
WHERE 
    a.user_id IS NULL 
GROUP BY 
    DATE_TRUNC('month', a.date)
```

Note that there are solutions to this problem that can use `LEFT` or `RIGHT` joins. 

### Part 3:

**Context**: You now want to see the number of active users this month *who have been reactivated* — in other words, users who have churned but this month they became active again. Keep in mind a user can reactivate after churning *before* the previous month. An example of this could be a user active in February (appears in `logins`), no activity in March and April, but then active again in May (appears in `logins`), so they count as a reactivated user for May . 

**Task:** Create a table that contains the number of reactivated users per month. 
* * *
***Solution:***

```
     SELECT 
        DATE_TRUNC('month', a.date) month_timestamp,
        COUNT(DISTINCT a.user_id) reactivated_users,
        /* 
        * At least in the flavors of SQL I have used, you don't need to 
        * include the columns used in HAVING in the SELECT statement.
        * I have written them out for clarity here.  
        */ 
        MAX(DATE_TRUNC('month', b.date)) most_recent_active_previously 
     FROM 
        logins a
     JOIN
        logins b ON a.user_id = b.user_id 
                AND 
                DATE_TRUNC('month', a.date) > DATE_TRUNC('month', b.date)
     GROUP BY 
        DATE_TRUNC('month', a.date)
     HAVING 
        month_timestamp > most_recent_active_previously + interval '1 month' 
```



## #4: Cumulative Sums 

**Acknowledgement:** This problem was inspired by Sisense’s[“Cash Flow modeling in SQL”](https://www.sisense.com/blog/cash-flow-modeling-in-sql/)
blog post 

**Context:** Say we have a table `transactions` in the form:

```
| date       | cash_flow |
|------------|-----------|
| 2018-01-01 | -1000     |
| 2018-01-02 | -100      |
| 2018-01-03 | 50        |
| ...        | ...       |
```

Where `cash_flow` is the revenues minus costs for each day. 

**Task: **Write a query to get *cumulative* cash flow for each day such that we end up with a table in the form below: 

```
| date       | cumulative_cf |
|------------|---------------|
| 2018-01-01 | -1000         |
| 2018-01-02 | -1100         |
| 2018-01-03 | -1050         |
| ...        | ...           |
```

* * *
***Solution:***

```
SELECT 
    a.date date, 
    SUM(b.cash_flow) as cumulative_cf 
FROM
    transactions a
JOIN b 
    transactions b ON a.date >= b.date 
GROUP BY 
    a.date 
ORDER BY 
    date ASC
```

Alternate solution using a window function (more efficient!):  

```
SELECT 
    date, 
    SUM(cash_flow) OVER (ORDER BY date ASC) as cumulative_cf 
FROM
    transactions 
ORDER BY 
    date ASC
```

## #5: Rolling Averages 

**Acknowledgement:** This problem is adapted from Sisense’s [“Rolling Averages in MySQL and SQL Server”](https://www.sisense.com/blog/rolling-average/) blog post 

**Note:** there are different ways to compute rolling/moving averages. Here we'll use a preceding average which means that the metric for the 7th day of the month would be the average of the preceding 6 days and that day itself. 

**Context**: Say we have table `signups` in the form: 

```
| date       | sign_ups |
|------------|----------|
| 2018-01-01 | 10       |
| 2018-01-02 | 20       |
| 2018-01-03 | 50       |
| ...        | ...      |
| 2018-10-01 | 35       |
```

**Task**: Write a query to get 7-day rolling (preceding) average of daily sign ups. 
* * *
***Solution:***

```
SELECT 
  a.date, 
  AVG(b.sign_ups) average_sign_ups 
FROM 
  signups a 
JOIN 
  signups b ON a.date <= b.date + interval '6 days' AND a.date >= b.date
GROUP BY 
  a.date
```



## #6: Multiple Join Conditions 

**Acknowledgement:** This problem was inspired by Sisense’s [“Analyzing Your Email with SQL”](https://www.sisense.com/blog/analyzing-your-email-with-sql/) blog post 

**Context:** Say we have a table `emails` that includes emails sent to and from [`zach@g.com`](mailto:zach@g.com):

```
| id | subject  | from         | to           | timestamp           |
|----|----------|--------------|--------------|---------------------|
| 1  | Yosemite | zach@g.com   | thomas@g.com | 2018-01-02 12:45:03 |
| 2  | Big Sur  | sarah@g.com  | thomas@g.com | 2018-01-02 16:30:01 |
| 3  | Yosemite | thomas@g.com | zach@g.com   | 2018-01-02 16:35:04 |
| 4  | Running  | jill@g.com   | zach@g.com   | 2018-01-03 08:12:45 |
| 5  | Yosemite | zach@g.com   | thomas@g.com | 2018-01-03 14:02:01 |
| 6  | Yosemite | thomas@g.com | zach@g.com   | 2018-01-03 15:01:05 |
| .. | ..       | ..           | ..           | ..                  |
```

**Task: **Write a query to get the response time per email (`id`) sent to `zach@g.com` . 
Do not include `id`s that did not receive a response from [zach@g.com](mailto:zach@g.com). Assume 
each email thread has a unique subject. Keep in mind a thread may have multiple responses 
back-and-forth between [zach@g.com](mailto:zach@g.com) and another email address. 
* * *
***Solution:***

```
SELECT 
    a.id, 
    MIN(b.timestamp) - a.timestamp as time_to_respond 
FROM 
    emails a 
JOIN
    emails b 
        ON 
            b.subject = a.subject 
        AND 
            a.to = b.from
        AND 
            a.from = b.to 
        AND 
            a.timestamp < b.timestamp 
 WHERE 
    a.to = 'zach@g.com' 
 GROUP BY 
    a.id 
```



# Window Function Practice Problems 

## #1: Get the ID with the highest value 

**Context:** Say we have a table `salaries` with data on employee salary and department in the following format: 

```
  depname  | empno | salary |     
-----------+-------+--------+
 develop   |    11 |   5200 | 
 develop   |     7 |   4200 | 
 develop   |     9 |   4500 | 
 develop   |     8 |   6000 | 
 develop   |    10 |   5200 | 
 personnel |     5 |   3500 | 
 personnel |     2 |   3900 | 
 sales     |     3 |   4800 | 
 sales     |     1 |   5000 | 
 sales     |     4 |   4800 | 
```

**Task**: Write a query to get the `empno` with the highest salary. Make sure your solution can handle ties!
* * *
***Solution:***

```
WITH max_salary AS (
    SELECT 
        MAX(salary) max_salary
    FROM 
        `salaries
    )
SELECT 
    s.empno
FROM 
    `salaries s
JOIN 
    max_salary ms ON s.salary = ms.max_salary ``
```

Alternate solution using `RANK()`:

```
WITH sal_rank AS 
  (SELECT 
    empno, 
    RANK() OVER(ORDER BY salary DESC) rnk
  FROM 
    salaries)
SELECT 
  empno
FROM
  sal_rank
WHERE 
  rnk = 1;
```



## #2: Average and rank with a window function (multi-part)

### Part 1: 

**Context**: Say we have a table `salaries` in the format:

```
  depname  | empno | salary |     
-----------+-------+--------+
 develop   |    11 |   5200 | 
 develop   |     7 |   4200 | 
 develop   |     9 |   4500 | 
 develop   |     8 |   6000 | 
 develop   |    10 |   5200 | 
 personnel |     5 |   3500 | 
 personnel |     2 |   3900 | 
 sales     |     3 |   4800 | 
 sales     |     1 |   5000 | 
 sales     |     4 |   4800 | 
```

**Task:** Write a query that returns the same table, but with a new column that has average salary per `depname`. We would expect a table in the form: 

```
  depname  | empno | salary | avg_salary |     
-----------+-------+--------+------------+
 develop   |    11 |   5200 |       5020 |
 develop   |     7 |   4200 |       5020 | 
 develop   |     9 |   4500 |       5020 |
 develop   |     8 |   6000 |       5020 | 
 develop   |    10 |   5200 |       5020 | 
 personnel |     5 |   3500 |       3700 |
 personnel |     2 |   3900 |       3700 |
 sales     |     3 |   4800 |       4867 | 
 sales     |     1 |   5000 |       4867 | 
 sales     |     4 |   4800 |       4867 |
```

* * *
***Solution:***

```
SELECT 
    *, 
    /*
    * AVG() is a Postgres command, but other SQL flavors like BigQuery use 
    * AVERAGE()
    */ 
    ROUND(AVG(salary),0) OVER (PARTITION BY depname) avg_salary
FROM
    salaries
```

### Part 2:

**Task:** Write a query that adds a column with the rank of each employee based on their salary within their department, where the employee with the highest salary gets the rank of `1`. We would expect a table in the form: 

```
  depname  | empno | salary | salary_rank |     
-----------+-------+--------+-------------+
 develop   |    11 |   5200 |           2 |
 develop   |     7 |   4200 |           5 | 
 develop   |     9 |   4500 |           4 |
 develop   |     8 |   6000 |           1 | 
 develop   |    10 |   5200 |           2 | 
 personnel |     5 |   3500 |           2 |
 personnel |     2 |   3900 |           1 |
 sales     |     3 |   4800 |           2 | 
 sales     |     1 |   5000 |           1 | 
 sales     |     4 |   4800 |           2 | 
```

* * *
***Solution:***

```
SELECT 
    *, 
    RANK() OVER(PARTITION BY depname ORDER BY salary DESC) salary_rank
 FROM  
    salaries 
```



# Other Medium/Hard SQL Practice Problems 

## #1: Histograms 

**Context:** Say we have a table `sessions` where each row is a video streaming session with length in seconds: 

```
| session_id | length_seconds |
|------------|----------------|
| 1          | 23             |
| 2          | 453            |
| 3          | 27             |
| ..         | ..             |
```

**Task:** Write a query to count the number of sessions that fall into bands of size 5, i.e. for the above snippet, produce something akin to: 

```
| bucket  | count |
|---------|-------|
| 20-25   | 2     |
| 450-455 | 1     |
```

Get complete credit for the proper string labels (“5-10”, etc.) but near complete credit for something that is communicable as the bin. 
* * *
***Solution:***

```
WITH bin_label AS 
(SELECT 
    session_id, 
    FLOOR(length_seconds/5) as bin_label 
 FROM
    sessions 
 )
 SELECT 
    `CONCATENTATE(STR(bin_label*5), '-', STR(`bin_label*5+5)) bucket, 
    COUNT(DISTINCT session_id) count ``
 GROUP BY 
    bin_label
 ORDER BY 
    `bin_label ASC `
```



## #2: CROSS JOIN (multi-part)

### Part 1: 

**Context:** Say we have a table `state_streams` where each row is a state and the total number of hours of streaming from a video hosting service: 

```
| state | total_streams |
|-------|---------------|
| NC    | 34569         |
| SC    | 33999         |
| CA    | 98324         |
| MA    | 19345         |
| ..    | ..            |
```

(In reality these kinds of aggregate tables would normally have a date column, but we’ll exclude that component in this problem) 

**Task:** Write a query to get the pairs of states with total streaming amounts within 1000 of each other. For the snippet above, we would want to see something like:

```
| state_a | state_b |
|---------|---------|
| NC      | SC      |
| SC      | NC      |
```

* * *
***Solution:***

```
SELECT
    a.state as state_a, 
    b.state as state_b 
 FROM   
    state_streams a
 CROSS JOIN 
    state_streams b 
 WHERE 
    ABS(a.total_streams - b.total_streams) < 1000
    AND 
    a.state <> b.state 
```

FYI, `CROSS JOIN` s can also be written without explicitly specifying a join: 

```
SELECT
    a.state as state_a, 
    b.state as state_b 
 FROM   
    state_streams a, state_streams b 
 WHERE 
    ABS(a.total_streams - b.total_streams) < 1000
    AND 
    a.state <> b.state 
```



### Part 2: 

**Note:** This question is considered more of a bonus problem than an actual SQL pattern. Feel free to skip it!

**Task:** How could you modify the SQL from the solution to Part 1 of this question so that duplicates are removed? For example, if we used the sample table from Part 1, the pair `NC` and `SC` should only appear in one row instead of two. 
* * *
***Solution: ***

```
SELECT
    a.state as state_a, 
    b.state as state_b 
 FROM   
    state_streams a, state_streams b 
 WHERE 
    ABS(a.total_streams - b.total_streams) < 1000
    AND 
    a.state > b.state 
```



## #3: Advancing Counting 

**Acknowledgement:** This question is adapted from [this Stack Overflow question](https://stackoverflow.com/questions/54488894/using-case-to-properly-count-items-with-if-else-logic-in-sql) by me (zthomas.nc) 

**Note:** this question is probably more complex than the kind you would encounter in an interview. Consider it a challenge problem, or feel free to skip it! 

**Context: **Say I have a table `table` in the following form, where a `user` can be mapped to multiple values of `class`:

```
| user | class |
|------|-------|
| 1    | a     |
| 1    | b     |
| 1    | b     |
| 2    | b     |
| 3    | a     |
```

**Task:** Assume there are only two possible values for `class`. Write a query to count the number of users in each class such that any user who has label `a` and `b` gets sorted into `b`, any user with just `a` gets sorted into `a` and any user with just `b` gets into `b`. 

For `table` that would result in the following table: 

```
| class | count |
|-------|-------|
| a     | 1     |
 | b     | 2     |
```

* * *
***Solution: ***

```
WITH usr_b_sum AS 
(
    SELECT 
        user, 
        SUM(CASE WHEN class = 'b' THEN 1 ELSE 0 END) num_b
    FROM 
        table
    GROUP BY 
        user
), 

usr_class_label AS 
(
    SELECT 
        user, 
        CASE WHEN num_b > 0 THEN 'b' ELSE 'a' END class 
    FROM 
        usr_b_sum
)

SELECT 
    class, 
    COUNT(DISTINCT user) count 
FROM
    usr_class_label
GROUP BY 
    class 
ORDER BY 
    class ASC

    
```

Alternate solution: Using `SELECT`s in the `SELECT` statement and `UNION`: 

```
SELECT 
    "a" class,
    COUNT(DISTINCT user_id) - 
        (SELECT COUNT(DISTINCT user_id) FROM table WHERE class = 'b') count 
UNION
SELECT 
    "b" class,
    (SELECT COUNT(DISTINCT user_id) FROM table WHERE class = 'b') count 
```

Alternate solution: Since the problem as stated didn’t ask for generalizable solution, you can leverage the fact that `b` > `a` to produce this straightforward solution: 

**Acknowledgement**: Thanks to Karan Gadiya for contributing this solution. Thanks, KG! 

```
WITH max_class AS (
    SELECT
        user, 
        MAX(class) as class 
    FROM 
        table 
    GROUP BY 
        user
)

SELECT 
    class, 
    COUNT(user)
FROM
    max_class
GROUP BY 
    class    
```
################################################################        
################################################################################################

Some useful stuff from: https://gvwilson.github.io/sql-tutorial/core/

Existence and Correlated Subqueries
select
    name,
    building
from department
where
    exists (
        select 1
        from staff
        where dept = department.ident
    )
order by name;
|       name        |     building     |
|-------------------|------------------|
| Genetics          | Chesson          |
| Histology         | Fashet Extension |
| Molecular Biology | Chesson          |
Endocrinology is missing from the list
select 1 could equally be select true or any other value
A correlated subquery depends on a value from the outer query
Equivalent to nested loop
Nonexistence
select
    name,
    building
from department
where
    not exists (
        select 1
        from staff
        where dept = department.ident
    )
order by name;
|     name      | building |
|---------------|----------|
| Endocrinology | TGVH     |


Avoiding Correlated Subqueries
select distinct
    department.name as name,
    department.building as building
from department inner join staff
    on department.ident = staff.dept
order by name;
|       name        |     building     |
|-------------------|------------------|
| Genetics          | Chesson          |
| Histology         | Fashet Extension |
| Molecular Biology | Chesson          |
The join might or might not be faster than the correlated subquery
Hard to find unstaffed departments without either not exists or count and a check for 0

Lead and Lag
with ym_num as (
    select
        strftime('%Y-%m', started) as ym,
        count(*) as num
    from experiment
    group by ym
)

select
    ym,
    lag(num) over (order by ym) as prev_num,
    num,
    lead(num) over (order by ym) as next_num
from ym_num
order by ym;


|   ym    | prev_num | num | next_num |
|---------|----------|-----|----------|
| 2023-01 |          | 2   | 5        |
| 2023-02 | 2        | 5   | 5        |
| 2023-03 | 5        | 5   | 1        |
| 2023-04 | 5        | 1   | 6        |
| 2023-05 | 1        | 6   | 5        |
| 2023-06 | 6        | 5   | 3        |
| 2023-07 | 5        | 3   | 2        |
| 2023-08 | 3        | 2   | 4        |
| 2023-09 | 2        | 4   | 6        |
| 2023-10 | 4        | 6   | 4        |
| 2023-12 | 6        | 4   | 5        |
| 2024-01 | 4        | 5   | 2        |
| 2024-02 | 5        | 2   |          |
Use strftime to extract year and month
Clumsy, but date/time handling is not SQLite’s strong point
Use window functions lead and lag to shift values
Unavailable values at the top or bottom are null


Boundaries
Documentation on SQLite’s window functions describes three frame types and five kinds of frame boundary
It feels very ad hoc, but so does the real world
Windowing Functions
with ym_num as (
    select
        strftime('%Y-%m', started) as ym,
        count(*) as num
    from experiment
    group by ym
)

select
    ym,
    num,
    sum(num) over (order by ym) as num_done,
    (sum(num) over (order by ym) * 1.00) / (select sum(num) from ym_num) as completed_progress,
    cume_dist() over (order by ym) as linear_progress
from ym_num
order by ym;
|   ym    | num | num_done | completed_progress |  linear_progress   |
|---------|-----|----------|--------------------|--------------------|
| 2023-01 | 2   | 2        | 0.04               | 0.0769230769230769 |
| 2023-02 | 5   | 7        | 0.14               | 0.153846153846154  |
| 2023-03 | 5   | 12       | 0.24               | 0.230769230769231  |
| 2023-04 | 1   | 13       | 0.26               | 0.307692307692308  |
| 2023-05 | 6   | 19       | 0.38               | 0.384615384615385  |
| 2023-06 | 5   | 24       | 0.48               | 0.461538461538462  |
| 2023-07 | 3   | 27       | 0.54               | 0.538461538461538  |
| 2023-08 | 2   | 29       | 0.58               | 0.615384615384615  |
| 2023-09 | 4   | 33       | 0.66               | 0.692307692307692  |
| 2023-10 | 6   | 39       | 0.78               | 0.769230769230769  |
| 2023-12 | 4   | 43       | 0.86               | 0.846153846153846  |
| 2024-01 | 5   | 48       | 0.96               | 0.923076923076923  |
| 2024-02 | 2   | 50       | 1.0                | 1.0                |
sum() over does a running total
cume_dist() is fraction of rows seen so far
So num_done column is number of experiments done…
…completed_progress is the fraction of experiments done…
…and linear_progress is the fraction of time passed


Partitioned Windows
with y_m_num as (
    select
        strftime('%Y', started) as year,
        strftime('%m', started) as month,
        count(*) as num
    from experiment
    group by year, month
)

select
    year,
    month,
    num,
    sum(num) over (partition by year order by month) as num_done
from y_m_num
order by year, month;
| year | month | num | num_done |
|------|-------|-----|----------|
| 2023 | 01    | 2   | 2        |
| 2023 | 02    | 5   | 7        |
| 2023 | 03    | 5   | 12       |
| 2023 | 04    | 1   | 13       |
| 2023 | 05    | 6   | 19       |
| 2023 | 06    | 5   | 24       |
| 2023 | 07    | 3   | 27       |
| 2023 | 08    | 2   | 29       |
| 2023 | 09    | 4   | 33       |
| 2023 | 10    | 6   | 39       |
| 2023 | 12    | 4   | 43       |
| 2024 | 01    | 5   | 5        |
| 2024 | 02    | 2   | 7        |
partition by creates groups
So this counts experiments started since the beginning of each year






Enumerating Rows
Every table has a special column called rowid
select
    rowid,
    species,
    island
from penguins
limit 5;
| rowid | species |  island   |
|-------|---------|-----------|
| 1     | Adelie  | Torgersen |
| 2     | Adelie  | Torgersen |
| 3     | Adelie  | Torgersen |
| 4     | Adelie  | Torgersen |
| 5     | Adelie  | Torgersen |
rowid is persistent within a session
I.e., if we delete the first 5 rows we now have row IDs 6…N
Do not rely on row ID


In particular, do not use it as a key

Conditionals

with sized_penguins as (
    select
        species,
        iif(
            body_mass_g < 3500,
            'small',
            'large'
        ) as size
    from penguins
    where body_mass_g is not null
)

select
    species,
    size,
    count(*) as num
from sized_penguins
group by species, size
order by species, num;
|  species  | size  | num |
|-----------|-------|-----|
| Adelie    | small | 54  |
| Adelie    | large | 97  |
| Chinstrap | small | 17  |
| Chinstrap | large | 51  |
| Gentoo    | large | 123 |
iif(condition, true_result, false_result)
Note: iif with two i’s
May feel odd to think of if/else as a function, but common in vectorized calculations

Selecting a Case
What if we want small, medium, and large?
Can nest iif, but quickly becomes unreadable
with sized_penguins as (
    select
        species,
        case
            when body_mass_g < 3500 then 'small'
            when body_mass_g < 5000 then 'medium'
            else 'large'
        end as size
    from penguins
    where body_mass_g is not null
)

select
    species,
    size,
    count(*) as num
from sized_penguins
group by species, size
order by species, num;
|  species  |  size  | num |
|-----------|--------|-----|
| Adelie    | small  | 54  |
| Adelie    | medium | 97  |
| Chinstrap | small  | 17  |
| Chinstrap | medium | 51  |
| Gentoo    | medium | 56  |
| Gentoo    | large  | 67  |
Evaluate when options in order and take first
Result of case is null if no condition is true
Use else as fallback


Checking a Range
with sized_penguins as (
    select
        species,
        case
            when body_mass_g between 3500 and 5000 then 'normal'
            else 'abnormal'
        end as size
    from penguins
    where body_mass_g is not null
)

select
    species,
    size,
    count(*) as num
from sized_penguins
group by species, size
order by species, num;
|  species  |   size   | num |
|-----------|----------|-----|
| Adelie    | abnormal | 54  |
| Adelie    | normal   | 97  |
| Chinstrap | abnormal | 17  |
| Chinstrap | normal   | 51  |
| Gentoo    | abnormal | 61  |
| Gentoo    | normal   | 62  |
between can make queries easier to read
But be careful of the and in the middle


Pattern Matching
select
    personal,
    family
from staff
where personal like '%ya%';


Selecting First and Last Rows
select * from (
    select * from (select * from experiment order by started asc limit 5)
    union all
    select * from (select * from experiment order by started desc limit 5)
)
order by started asc;
| ident |    kind     |  started   |   ended    |
|-------|-------------|------------|------------|
| 17    | trial       | 2023-01-29 | 2023-01-30 |
| 35    | calibration | 2023-01-30 | 2023-01-30 |
| 36    | trial       | 2023-02-02 | 2023-02-03 |
| 25    | trial       | 2023-02-12 | 2023-02-14 |
| 2     | calibration | 2023-02-14 | 2023-02-14 |
| 40    | calibration | 2024-01-21 | 2024-01-21 |
| 12    | trial       | 2024-01-26 | 2024-01-28 |
| 44    | trial       | 2024-01-27 | 2024-01-29 |
| 34    | trial       | 2024-02-01 | 2024-02-02 |
| 14    | calibration | 2024-02-03 | 2024-02-03 |
union all combines records
Keeps duplicates: union on its own only keeps unique records
Which is more work but sometimes more useful
Yes, it feels like the extra select * from should be unnecessary


Intersection
select
    personal,
    family,
    dept,
    age
from staff
where dept = 'mb'
intersect
select
    personal,
    family,
    dept,
    age from staff
where age < 50;
| personal |  family   | dept | age |
|----------|-----------|------|-----|
| Indrans  | Sridhar   | mb   | 47  |
| Ishaan   | Ramaswamy | mb   | 35  |
Rows involved must have the same structure
Intersection usually used when pulling values from different sources
In the query above, would be clearer to use where

Random Numbers and Why Not
with decorated as (
    select random() as rand,
    personal || ' ' || family as name
    from staff
)

select
    rand,
    abs(rand) % 10 as selector,
    name
from decorated
where selector < 5;
|         rand         | selector |      name       |
|----------------------|----------|-----------------|
| -5088363674211922423 | 0        | Divit Dhaliwal  |
| 6557666280550701355  | 1        | Indrans Sridhar |
| -2149788664940846734 | 3        | Pranay Khanna   |
| -3941247926715736890 | 8        | Riaan Dua       |
| -3101076015498625604 | 5        | Vedika Rout     |
| -7884339441528700576 | 4        | Abram Chokshi   |
| -2718521057113461678 | 4        | Romil Kapoor    |
There is no way to seed SQLite’s random number generator
Which means there is no way to reproduce its pseudo-random sequences
Which means you should never use it
How are you going to debug something you can’t re-run?

Generating Sequences
select value from generate_series(1, 5);
| value |
|-------|
| 1     |
| 2     |
| 3     |
| 4     |
| 5     |


Self Join
with person as (
    select
        ident,
        personal || ' ' || family as name
    from staff
)

select
    left_person.name,
    right_person.name
from person as left_person inner join person as right_person
limit 10;
|     name     |       name       |
|--------------|------------------|
| Kartik Gupta | Kartik Gupta     |
| Kartik Gupta | Divit Dhaliwal   |
| Kartik Gupta | Indrans Sridhar  |
| Kartik Gupta | Pranay Khanna    |
| Kartik Gupta | Riaan Dua        |
| Kartik Gupta | Vedika Rout      |
| Kartik Gupta | Abram Chokshi    |
| Kartik Gupta | Romil Kapoor     |
| Kartik Gupta | Ishaan Ramaswamy |
| Kartik Gupta | Nitya Lal        |
Join a table to itself
Use as to create aliases for copies of tables to distinguish them
Nothing special about the names left and right
Get all 
 pairs, including person with themself


Generating Unique Pairs
with person as (
    select
        ident,
        personal || ' ' || family as name
    from staff
)

select
    left_person.name,
    right_person.name
from person as left_person inner join person as right_person
on left_person.ident < right_person.ident
where left_person.ident <= 4 and right_person.ident <= 4;
|      name       |      name       |
|-----------------|-----------------|
| Kartik Gupta    | Divit Dhaliwal  |
| Kartik Gupta    | Indrans Sridhar |
| Kartik Gupta    | Pranay Khanna   |
| Divit Dhaliwal  | Indrans Sridhar |
| Divit Dhaliwal  | Pranay Khanna   |
| Indrans Sridhar | Pranay Khanna   |
left.ident < right.ident ensures distinct pairs without duplicates
Query uses left.ident <= 4 and right.ident <= 4 to shorten output
Quick check: 
 pairs

Filtering Pairs
with
person as (
    select
        ident,
        personal || ' ' || family as name
    from staff
),

together as (
    select
        left_perf.staff as left_staff,
        right_perf.staff as right_staff
    from performed as left_perf inner join performed as right_perf
        on left_perf.experiment = right_perf.experiment
    where left_staff < right_staff
)

select
    left_person.name as person_1,
    right_person.name as person_2
from person as left_person inner join person as right_person join together
    on left_person.ident = left_staff and right_person.ident = right_staff;
|    person_1     |     person_2     |
|-----------------|------------------|
| Kartik Gupta    | Vedika Rout      |
| Pranay Khanna   | Vedika Rout      |
| Indrans Sridhar | Romil Kapoor     |
| Abram Chokshi   | Ishaan Ramaswamy |
| Pranay Khanna   | Vedika Rout      |
| Kartik Gupta    | Abram Chokshi    |
| Abram Chokshi   | Romil Kapoor     |
| Kartik Gupta    | Divit Dhaliwal   |
| Divit Dhaliwal  | Abram Chokshi    |
| Pranay Khanna   | Ishaan Ramaswamy |
| Indrans Sridhar | Romil Kapoor     |
| Kartik Gupta    | Ishaan Ramaswamy |
| Kartik Gupta    | Nitya Lal        |
| Kartik Gupta    | Abram Chokshi    |
| Pranay Khanna   | Romil Kapoor     |







Set Membership
select *
from work
where person not in ('mik', 'tay');
| person |   job    |
|--------|----------|
| po     | clean    |
| po     | complain |
in values and not in values do exactly what you expect


Subqueries
select distinct person
from work
where person not in (
    select distinct person
    from work
    where job = 'calibrate'
);
| person |
|--------|
| po     |
| tay    |
Use a subquery to select the people who do calibrate
Then select all the people who aren’t in that set
Initially feels odd, but subqueries are useful in other ways


Creating New Tables from Old
create table new_work (
    person_id integer not null,
    job_id integer not null,
    foreign key (person_id) references person (ident),
    foreign key (job_id) references job (ident)
);

insert into new_work
select
    person.ident as person_id,
    job.ident as job_id
from
    (person inner join work on person.name = work.person)
    inner join job on job.name = work.job;
select * from new_work;
| person_id | job_id |
|-----------|--------|
| 1         | 1      |
| 1         | 2      |
| 2         | 2      |
new_work is our join table
Each column refers to a record in some other table



Comparing Individual Values to Aggregates
Go back to the original penguins database
select body_mass_g
from penguins
where
    body_mass_g > (
        select avg(body_mass_g)
        from penguins
    )
limit 5;
| body_mass_g |
|-------------|
| 4675.0      |
| 4250.0      |
| 4400.0      |
| 4500.0      |
| 4650.0      |
Get average body mass in subquery
Compare each row against that
Requires two scans of the data, but no way to avoid that
Except calculating a running total each time a penguin is added to the table
Null values aren’t included in the average or in the final results


Comparing Individual Values to Aggregates Within Groups
select
    penguins.species,
    penguins.body_mass_g,
    round(averaged.avg_mass_g, 1) as avg_mass_g
from penguins inner join (
    select
        species,
        avg(body_mass_g) as avg_mass_g
    from penguins
    group by species
) as averaged
    on penguins.species = averaged.species
where penguins.body_mass_g > averaged.avg_mass_g
limit 5;
| species | body_mass_g | avg_mass_g |
|---------|-------------|------------|
| Adelie  | 3750.0      | 3700.7     |
| Adelie  | 3800.0      | 3700.7     |
| Adelie  | 4675.0      | 3700.7     |
| Adelie  | 4250.0      | 3700.7     |
| Adelie  | 3800.0      | 3700.7     |
Subquery runs first to create temporary table averaged with average mass per species
Join that with penguins
Filter to find penguins heavier than average within their species


Common Table Expressions
with grouped as (
    select
        species,
        avg(body_mass_g) as avg_mass_g
    from penguins
    group by species
)

select
    penguins.species,
    penguins.body_mass_g,
    round(grouped.avg_mass_g, 1) as avg_mass_g
from penguins inner join grouped
where penguins.body_mass_g > grouped.avg_mass_g
limit 5;
| species | body_mass_g | avg_mass_g |
|---------|-------------|------------|
| Adelie  | 3750.0      | 3700.7     |
| Adelie  | 3800.0      | 3700.7     |
| Adelie  | 4675.0      | 3700.7     |
| Adelie  | 4250.0      | 3700.7     |
| Adelie  | 3800.0      | 3700.7     |
Use common table expression (CTE) to make queries clearer
Nested subqueries quickly become difficult to understand
Database decides how to optimize


Explaining Query Plans
explain query plan
select
    species,
    avg(body_mass_g)
from penguins
group by species;
QUERY PLAN
|--SCAN penguins
`--USE TEMP B-TREE FOR GROUP BY
SQLite plans to scan every row of the table
It will build a temporary B-tree data structure to group rows













Removing Duplicates
select distinct
    species,
    sex,
    island
from penguins;
|  species  |  sex   |  island   |
|-----------|--------|-----------|
| Adelie    | MALE   | Torgersen |
| Adelie    | FEMALE | Torgersen |
| Adelie    |        | Torgersen |
| Adelie    | FEMALE | Biscoe    |
| Adelie    | MALE   | Biscoe    |
| Adelie    | FEMALE | Dream     |
| Adelie    | MALE   | Dream     |
| Adelie    |        | Dream     |
| Chinstrap | FEMALE | Dream     |
| Chinstrap | MALE   | Dream     |
| Gentoo    | FEMALE | Biscoe    |
| Gentoo    | MALE   | Biscoe    |
| Gentoo    |        | Biscoe    |
distinct keyword must appear right after select
SQL was supposed to read like English
Shows distinct combinations
Blanks in sex column show missing data
We’ll talk about this in a bit


Doing Calculations
select
    flipper_length_mm / 10.0,
    body_mass_g / 1000.0
from penguins
limit 3;
| flipper_length_mm / 10.0 | body_mass_g / 1000.0 |
|--------------------------|----------------------|
| 18.1                     | 3.75                 |
| 18.6                     | 3.8                  |
| 19.5                     | 3.25                 |
Can do the usual kinds of arithmetic on individual values
Calculation done for each row independently
Column name shows the calculation done


Calculating with Missing Values
select
    flipper_length_mm / 10.0 as flipper_cm,
    body_mass_g / 1000.0 as weight_kg,
    island as where_found
from penguins
limit 5;
| flipper_cm | weight_kg | where_found |
|------------|-----------|-------------|
| 18.1       | 3.75      | Torgersen   |
| 18.6       | 3.8       | Torgersen   |
| 19.5       | 3.25      | Torgersen   |
|            |           | Torgersen   |
| 19.3       | 3.45      | Torgersen   |
SQL uses a special value null to representing missing data
Not 0 or empty string, but “I don’t know”
Flipper length and body weight not known for one of the first five penguins
“I don’t know” divided by 10 or 1000 is “I don’t know”


Handling Null Safely
select
    species,
    sex,
    island
from penguins
where sex is null;
| species | sex |  island   |
|---------|-----|-----------|
| Adelie  |     | Torgersen |
| Adelie  |     | Torgersen |
| Adelie  |     | Torgersen |
| Adelie  |     | Torgersen |
| Adelie  |     | Torgersen |
| Adelie  |     | Dream     |
| Gentoo  |     | Biscoe    |
| Gentoo  |     | Biscoe    |
| Gentoo  |     | Biscoe    |
| Gentoo  |     | Biscoe    |
| Gentoo  |     | Biscoe    |
Use is null and is not null to handle null safely
Other parts of SQL handle nulls specially



Counting
select
    count(*) as count_star,
    count(sex) as count_specific,
    count(distinct sex) as count_distinct
from penguins;
| count_star | count_specific | count_distinct |
|------------|----------------|----------------|
| 344        | 333            | 2              |
count(*) counts rows
count(column) counts non-null entries in column
count(distinct column) counts distinct non-null entries

Behavior of Unaggregated Columns
select
    sex,
    avg(body_mass_g) as average_mass_g
from penguins
group by sex;
|  sex   |  average_mass_g  |
|--------|------------------|
|        | 4005.55555555556 |
| FEMALE | 3862.27272727273 |
| MALE   | 4545.68452380952 |
All rows in each group have the same value for sex, so no need to aggregate
Arbitrary Choice in Aggregation
select
    sex,
    body_mass_g
from penguins
group by sex;
|  sex   | body_mass_g |
|--------|-------------|
|        |             |
| FEMALE | 3800.0      |
| MALE   | 3750.0      |
If we don’t specify how to aggregate a column, SQLite chooses any arbitrary value from the group
All penguins in each group have the same sex because we grouped by that, so we get the right answer
The body mass values are in the data but unpredictable
A common mistake
Other database managers don’t do this
E.g., PostgreSQL complains that column must be used in an aggregation function


Readable Output
select
    sex,
    round(avg(body_mass_g), 1) as average_mass_g
from penguins
group by sex
having average_mass_g > 4000.0;
| sex  | average_mass_g |
|------|----------------|
|      | 4005.6         |
| MALE | 4545.7         |
Use round(value, decimals) to round off a number


Filtering Aggregate Inputs
select
    sex,
    round(
        avg(body_mass_g) filter (where body_mass_g < 4000.0),
        1
    ) as average_mass_g
from penguins
group by sex;
|  sex   | average_mass_g |
|--------|----------------|
|        | 3362.5         |
| FEMALE | 3417.3         |
| MALE   | 3729.6         |
filter (where condition) applies to inputs



Combining Information
select *
from work cross join job;
| person |    job    |   name    | billable |
|--------|-----------|-----------|----------|
| mik    | calibrate | calibrate | 1.5      |
| mik    | calibrate | clean     | 0.5      |
| mik    | clean     | calibrate | 1.5      |
| mik    | clean     | clean     | 0.5      |
| mik    | complain  | calibrate | 1.5      |
| mik    | complain  | clean     | 0.5      |
| po     | clean     | calibrate | 1.5      |
| po     | clean     | clean     | 0.5      |
| po     | complain  | calibrate | 1.5      |
| po     | complain  | clean     | 0.5      |
| tay    | complain  | calibrate | 1.5      |
| tay    | complain  | clean     | 0.5      |
A join combines information from two tables
cross join constructs their cross product
All combinations of rows from each
Result isn’t particularly useful: job and name values don’t match
I.e., the combined data has records whose parts have nothing to do with each other
Inner Join
select *
from work inner join job
    on work.job = job.name;
| person |    job    |   name    | billable |
|--------|-----------|-----------|----------|
| mik    | calibrate | calibrate | 1.5      |
| mik    | clean     | clean     | 0.5      |
| po     | clean     | clean     | 0.5      |
Use table.column notation to specify columns
A column can have the same name as a table
Use on condition to specify join condition
Since complain doesn’t appear in job.name, none of those rows are kept



Aggregating Joined Data
select
    work.person,
    sum(job.billable) as pay
from work inner join job
    on work.job = job.name
group by work.person;
| person | pay |
|--------|-----|
| mik    | 2.0 |
| po     | 0.5 |
Combines ideas we’ve seen before
But Tay is missing from the table
No records in the job table with tay as name
So no records to be grouped and summed


A left outer join keeps all rows from the left table


Aggregating Left Joins
select
    work.person,
    sum(job.billable) as pay
from work left join job
    on work.job = job.name
group by work.person;
| person | pay |
|--------|-----|
| mik    | 2.0 |
| po     | 0.5 |
| tay    |     |
That’s better, but we’d like to see 0 rather than a blank

Coalescing Values
select
    work.person,
    coalesce(sum(job.billable), 0.0) as pay
from work left join job
    on work.job = job.name
group by work.person;
| person | pay |
|--------|-----|
| mik    | 2.0 |
| po     | 0.5 |
| tay    | 0.0 |
coalesce(val1, val2, …) returns first non-null value










########################################################################
########################################################################
########################################################################


What is DBMS?
A database management system (DBMS) is a set of tools that make it easier for users to construct and maintain databases. In other words, a database management system (DBMS) provides us with an interface or tool for completing various tasks such as creating a database, entering data into it, deleting data from it, updating data, and so on. A database management system (DBMS) is software that allows data to be kept in a more secure manner than a file-based system. We can solve a variety of issues using DBMS, such data redundancy, data inconsistency, quick access, more ordered and intelligible data, and so on.
There are some well-known Database Management Systems, such as MySQL, Oracle, SQL Server, Amazon Simple DB (Cloud-based), and so on.
DBMS

What is Database?
A database is a collection of logical, consistent, and organised data that can be easily accessed, controlled, and updated. Databases, also known as electronic databases, are structured to allow for the efficient production, insertion, and updating of data and are saved as a file or set of files on magnetic discs, tapes, and other secondary devices. Objects (tables) make up the majority of a database, and tables contain records and fields. Fields are the fundamental units of data storage, containing information on a certain element or attribute of the database's entity. A database management system (DBMS) is used to extract data from a database in the form of queries. To make it easier to access relevant information, you can organise data into tables, rows, and columns, as well as index it.
Database handlers design a database such that all users have access to the data through a single piece of tools.
The database's primary goal is to manage a huge amount of data by storing, retrieving, and managing it.
Databases are used to manage a large number of dynamic websites on the Internet today. Consider a model that checks the availability of hotel rooms. It's an example of a database-driven dynamic webpage. Databases such as MySQL, Sybase, Oracle, MongoDB, Informix, PostgreSQL, SQL Server, and others are available.

Mention the issues with traditional file-based systems that make DBMS a better choice?
The lack of indexing in a traditional file-based system leaves us with little choice but to scan the entire page, making content access time-consuming and slow. The other issue is redundancy and inconsistency, as files often include duplicate and redundant data, and updating one causes all of them to become inconsistent. Traditional file-based systems make it more difficult to access data since the data is disorganised.
Another drawback is the lack of concurrency management, which causes one operation to lock the entire page, as opposed to DBMS, which allows several operations to work on the same file at the same time.
Other concerns with traditional file-based systems that DBMSs have addressed include integrity checks, data isolation, atomicity, security, and so on.

Explain a few advantages of a DBMS.
The following are some of the benefits of using a database management system:

🚀 Data sharing: Data from the same database can be accessed by multiple people at the same time.
🚀 Integrity restrictions: These limitations allow for more refined data storage in a database.
🚀 Data redundancy control: Supports a system for controlling data redundancy by combining all data into a single database.
🚀 Data Independence: Allows the structure of the data to be changed without affecting the structure of any running application applications.
What is Java
🚀 Backup and recovery feature: Provides a 'backup and recovery' feature that automatically creates a data backup and restores the data as needed.
🚀 Data Security: A database management system (DBMS) provides the capabilities needed to make data storage and transfer more dependable and secure. Some common technologies used to safeguard data in a DBMS include authentication (the act of granting restricted access to a user) and encryption (encrypting sensitive data such as OTP, credit card information, and so on).

Explain different languages present in DBMS.
The following are some of the DBMS languages:
🚀 DDL (Data Definition Language) is a language that contains commands for defining databases. CREATE, ALTER, DROP, TRUNCATE, RENAME, and so on.
🚀 DML (Data Manipulation Language) is a set of commands that can be used to manipulate data in a database. SELECT, UPDATE, INSERT, DELETE, and so on.
What is Java
🚀 DCL (Data Control Language): It offers commands for dealing with the database system's user permissions and controls. GRANT and REVOKE, for example.
🚀 TCL (Transaction Control Language) is a programming language that offers commands for dealing with database transactions. COMMIT, ROLLBACK, and SAVEPOINT, for example.

What is meant by ACID properties in DBMS?
The ACID properties of a database management system are the basic principles that must be observed in order to maintain data integrity. They are as follows:
🚀Atomicity - Also known as the "all or nothing" rule, atomicity states that everything evaluated as a single unit is either executed to completion or not at all.
🚀 Consistency - This attribute indicates that the database's data is consistent before and after each transaction.
🚀What is Java
Isolation - This characteristic specifies that several transactions can be conducted at the same time.
🚀 Durability - This characteristic ensures that each transaction is saved in non-volatile memory once it has been finished.

Are NULL values in a database the same as that of blank space or zero?
No, a NULL value is distinct from zero and blank space in that it denotes a value that is assigned, unknown, unavailable, or not applicable, as opposed to blank space, which denotes a character, and zero, which denotes a number.
For instance, a NULL value in "number of courses" taken by a student indicates that the value is unknown, but a value of 0 indicates that the student has not taken any courses.

What are super, primary, candidate, and foreign keys?
🚀A super key is a set of relation schema attributes that all other schema attributes are functionally dependent on. The values of super key attributes cannot be the identical in any two rows.
🚀A Candidate key is a minimum superkey, which means that no suitable subset of Candidate key properties may be used to create a superkey.
🚀One of the candidate keys is the Primary Key.
🚀One of the candidate keys is chosen as the primary key and becomes the most important. In a table, there can only be one main key.
🚀A foreign key is a field (or set of fields) in one table that is used to uniquely identify a row in another table.

keys
9. What is the difference between primary key and unique constraints?
Although the primary key cannot have a NULL value, the unique constraints can. A table has just one main key, but it might have numerous unique constraints.

What is meant by DBMS and what is its utility? Explain RDBMS with examples.
The Database Management System, or DBMS, is a collection of applications or programmes that allow users to construct and maintain databases. A database management system (DBMS) provides a tool or interface for executing various database activities such as inserting, removing, updating, and so on. It is software that allows data to be stored in a more compact and secure manner than a file-based system. A database management system (DBMS) assists a user in overcoming issues such as data inconsistency, data redundancy, and other issues in a database, making it more comfortable and organised to use.
File systems, XML, the Windows Registry, and other DBMS systems are examples of prominent DBMS systems.
RDBMS stands for Relational Database Management System, and it was first introduced in the 1970s to make it easier to access and store data than DBMS. In contrast to DBMS, which stores data as files, RDBMS stores data as tables. When opposed to DBMS, storing data as rows and columns makes it easier to locate specific values in the database and makes it more efficient.
MySQL, Oracle DB, and other prominent RDBMS systems are examples.

What is a checkpoint in DBMS?
The Checkpoint is a technique that removes all previous logs from the system and stores them permanently on the storage drive.
Preserving the log of each transaction and maintaining shadow pages are two methods that can assist the DBMS in recovering and maintaining the ACID properties. When it comes to a log-based recovery system, checkpoints are necessary. Checkpoints are the minimal points from which the database engine can recover after a crash as a specified minimal point from which the transaction log record can be utilised to recover all committed data up to the moment of the crash.

What is a database system?
A database system is a collection of database and database management system software. We can execute a variety of tasks using the database system, including:
The data can be easily stored in the database, and there are no concerns about data redundancy or inconsistency.
When necessary, data will be pulled from the database using DBMS software. As a result, using database and DBMS software together allows you to store, retrieve, and access data with precision and security.

What do you mean by Data Model?
A data model consists of a set of tools for describing data, semantics, and constraints. They also assist in the description of the relationship between data entities and their attributes. Hierarchical data models, network models, entity relationship models, and relational models are some of the most prevalent data models. You may also learn more about data models by looking at other data modelling interview questions.

When does checkpoint occur in DBMS?
A checkpoint is a snapshot of the database management system's current state. The DBMS can use checkpoints to limit the amount of work that needs to be done during a restart in the event of a subsequent crash. After a system crash, checkpoints are utilised to recover the database. The log-based recovery solution employs checkpoints. When we need to restart the system because of a system crash, we use checkpoints. As a result, we won't have to execute the transactions from the beginning.

What is the difference between an entity and an attribute?
In a database, an entity is a real-world thing. Employee, designation, department, and so on are examples of different entities in an employee database.
A trait that describes an entity is called an attribute. For example, the entity "employee" can have properties such as name, ID, and age.

What are the various kinds of interactions catered by DBMS?
DBMS can handle a variety of interactions, including:

🚀Data definition
🚀 Update
🚀 Retrieval
🚀 Administration

What do you understand by query optimization?
Query optimization is the phase in which a plan for evaluating a query with the lowest estimated cost is identified. When there are numerous algorithms and approaches to perform the same goal, this phase emerges.
The following are some of the benefits of query optimization:
• The output is delivered more quickly.

• In less time, a higher number of queries can be run.

• Reduces the complexity of time and space

Do we consider NULL values the same as that of blank space or zero?
A NULL value is not to be confused with a value of zero or a blank space. While zero is a number and blank space is a character, NULL denotes a value that is unavailable, unknown, assigned, or not applicable.

What do you understand by aggregation and atomicity?

Aggregation Atomicity
This is an E-R model feature that allows one relationship set to interact with another relationship set. This attribute specifies that a database alteration must either adhere to all of the rules or not at all. As a result, if one portion of the transaction fails, the transaction as a whole fails.

What are the different levels of abstraction in the DBMS?
In DBMS, there are three degrees of data abstraction.
Abstraction
They are as follows:
🚀Physical Level : The physical level of abstraction specifies how data is stored and is the lowest degree of abstraction.
🚀Logical Layer : After the Physical level, there is the Logical level of abstraction. This layer decides what data is saved in the database and how the data pieces relate to one another.
🚀 View Level: The greatest level of abstraction, the View Level describes only a portion of the entire database.

What is an entity-relationship model?
It's a diagrammatic approach to database architecture in which real-world things are represented as entities and relationships between them are mentioned. This method allows the DBA staff to quickly grasp the schema.

What do you understand by the terms Entity, Entity Type, and Entity Set in DBMS?

🚀Entity: An entity is a real-world object with attributes, which are nothing more than the object's qualities. An employee, for example, is a type of entity. This entity can have attributes like empid, empname, and so on.
🚀Entity Type: An entity type is a collection of entities with similar attributes. An entity type, in general, refers to one or more related tables in a database. As a result, entity type can be thought of as a trait that uniquely identifies an entity. Employees can have attributes such as empid, empname, department, and so on.
🚀Entity Set: In a database, an entity set is a collection of all the entities of a specific entity type.An entity set can include, for example, a group of employees, a group of companies, and a group of persons.

What do you mean by transparent DBMS?
The transparent DBMS is a form of database management system that conceals its physical structure from users. Physical structure, also known as physical storage structure, refers to the DBMS's memory manager and explains how data is saved on disc.

What are the unary operations in Relational Algebra?
In relational algebra, the unary operations are PROJECTION and SELECTION. Single-operand operations are known as unary operations. SELECTION, PROJECTION, and RENAME are unary operations.
Relational operators, like as =,=,>=, and others, are employed in SELECTION.

What is RDBMS?
Relational Database Management Systems (RDBMS) is an acronym for Relational Database Management Systems. It's used to keep track of data records and table indices. RDBMS is a type of database management system that employs structure to identify and access data about other pieces of data in the database. RDBMS is a database management system that allows you to update, insert, delete, manipulate, and administer a relational database with minimal effort. The SQL language is utilised by RDBMS the majority of the time since it is simple to grasp and is frequently employed.

What are the differnt data models?
There are number of data modesl and they are :

🚀 Hierarchical data model
🚀Network model
🚀Relational model
🚀Entity-Relationship model

Define a Relation Schema and a Relation.
A Relation Schema is a collection of properties that define a relationship. Table schema is another name for it. It specifies the name of the table. The blueprint with which we may explain how data is grouped into tables is known as the relation schema. There is no data in this blueprint.
A set of tuples is used to define a relation. A connection is a collection of connected attributes with key attributes that identify them.
Consider the following scenario:
Let r be the relation containing set tuples (t1, t2, t3, ..., tn). Each tuple consists of an ordered list of n-values (t=1) (v1,v2, ...., vn).

What is Degree of relation?
The degree of a relationship is one of its relation schema's attributes. A degree of connection, also known as Cardinality, is defined as the number of times one entity occurs in relation to the number of times another entity occurs. One-to-one (1:1), one-to-many (1:M), and many-to-one (1:N) are the three degrees of relation (M:M).

What is Relationship?
An association between two or more entities is characterised as a relationship. In a database management system, there are three types of relationships:

🚀One-to-One: In this case, one record of any object can be linked to another object's record.
🚀One-to-Many (many-to-one): In this case, one record of any object can be linked to many records of other objects, and vice versa.
🚀Many-to-many: In this case, multiple records of one item can be linked to n records of another object.

What are the disadvantages of file processing systems?
tThe disadvantages of file processing systems are :
🚀 Data redundancy
🚀 Not secure
🚀 Inconsistent
🚀 Difficult in accessing data
🚀 Limited data sharing
🚀 Data integrity
🚀 Concurrent access is not possible
🚀 Data isolation
🚀 Atomicity problem

What is Data Abstraction in DBMS?
In a database management system, data abstraction is the process of hiding unimportant facts from users. Because database systems are made up of complicated data structures, user interaction with the database is made possible.
For example, we know that most users prefer systems with a simple graphical user interface (GUI), which means no sophisticated processing. As a result, data abstraction is required to keep the user engaged and to make data access simple. Furthermore, data abstraction divides the system into layers, allowing the job to be stated and properly defined.

Why is the use of DBMS recommended? Explain by listing some of its major advantages?
The following are some of the primary advantages of DBMS:

🚀Controlled Redundancy: DBMS enables a way to control data redundancy inside the database by integrating all data into a single database and preventing duplication of data because data is stored in just one location.
🚀Data Sharing: In a DBMS, data can be shared among several users at the same time because the same database is shared across all users and by various application applications.
🚀Backup and Recovery Facility: DBMS eliminates the burden of producing data backups over and over by including a 'backup and recovery' function that automatically produces data backups and restores them as needed.
🚀Integrity Constraints Must Be Enforced: Integrity Constraints must be enforced. So That The Refined Data Is Stored In The Database And This Is Followed By DBMS
🚀Data independence: It basically means that you can modify the data structure without affecting the structure of any application applications.

What is the difference between having and where clause?
In a select statement, HAVING is used to establish a condition for a group or an aggregate function. Before grouping, the WHERE clause picks. After grouping, the HAVING clause picks rows. The WHERE clause, unlike the HAVING clause, cannot contain aggregate functions.

What is a transaction? What are ACID properties?
A database transaction is a collection of database operations that must be handled as a whole, meaning that all or none of the actions must be executed. A bank transaction from one account to another is a good illustration. Either both debit and credit operations must be completed, or none of them must be completed. The ACID qualities (Atomicity, Consistency, Isolation, and Durability) ensure that database transactions are processed reliably.

What is Join?
An SQL Join is a technique for combining data from two or more tables based on a shared field.

What is Identity?
Identity (or AutoNumber) is a column that creates numeric values automatically. It is possible to set a start and increment value, however most DBAs leave them at 1. A GUID column generates numbers as well, but the value cannot be changed. There is no need to index the identity/GUID columns.

What is view in SQL?
A view is a virtual table created from a SQL statement's result set. We can use the create view syntax to do so.

What are uses of view?
The uses of view are as follows :

Views can represent a subset of the data in a table; as a result, a view can limit the extent to which the underlying tables are exposed to the outside world: a user may be allowed to query the view but not the whole of the base database.

Views allow you to combine and simplify numerous tables into a single virtual table.

Views can be used as aggregated tables, in which the database engine aggregates data (sum, average, and so on) and displays the generated results alongside the data.

Views can obscure data complexity.

Views take up extremely minimal storage space; the database simply maintains the specification of a view, not a copy of all the data it displays.

Views can give additional security depending on the SQL engine utilised.

What is a Trigger?
A trigger is a code that is connected with inserting, updating, or deleting data. When a table's associated query is run, the code is automatically performed. Triggers are useful for maintaining database integrity.

What is a stored procedure?
A stored procedure is similar to a function in that it contains a collection of operations that have been put together. It includes a set of procedures that are frequently used in applications to perform database activities.

What is the difference between Trigger and Stored Procedure?
Triggers, unlike Stored Procedures, cannot be called directly. Only inquiries can be linked to them.

What is database normalization?
It is a method for assessing relation schemas based on their functional dependencies and primary keys in order to obtain the following desirable properties:

Keeping Redundancy to a Minimum

Reducing Insertion, Deletion, and Update Inconsistencies
Relation schemas that don't meet the properties are broken down into smaller relation schemas that might meet the requirements.

What are indexes?
A database index is a data structure that improves the speed of data retrieval operations on a database table at the expense of more writes and storage space to keep the extra copy of data. On a disc, data can only be stored in one order. Faster search, such as binary search for different values, is sought to provide faster access according to different values. Indexes on tables are constructed for this purpose. These indexes take up more disc space, but they allow for speedier searches based on several frequently queried parameters.

What are clustered and non-clustered Indexes?
Clustered indexes are the indexes that determine how data is stored on a disc. As a result, each database table can only have one clustered index. Non-clustered indexes define logical ordering rather than physical ordering of data. In most cases, a tree is generated, with the leaves pointing to disc records. For this, a B-tree or B+ tree is utilised.

What is Denormalization?
Denormalization is a database optimization method in which duplicated data is added to one or more tables.

What is CLAUSE in SQL?
In SQL, a clause is a portion of a query that allows you to filter or personalise how your data is queried for you.

What is LiveLock?
When two or more processes repeatedly repeat the same interaction in reaction to changes in the other processes without producing any beneficial work, this is known as a livelock situation. These processes are not in a condition of waiting, and they are all executing at the same time. This is distinct from a stalemate, which occurs when all processes are in a state of waiting.

Livelock
48. What is QBE?
Query-by-example is a visual/graphical technique to obtaining information in a database by using skeleton tables as query templates. It's used to express what has to be done by explicitly entering example values into a query template. Many database systems for personal computers use QBE. QBE is a very strong tool that allows the user to access the information they want without having to know any programming languages. Skeleton tables are used to express queries in QBE. QBE has two distinguishing characteristics:
Queries in QBE use a two-dimensional syntax, which makes them look like tables.

Why are cursors necessary in embedded SQL?
A cursor is an object that stores the outcome of a query so that application programmes can process it row by row. SQL statements are statements that operate on a collection of data and return another set of data. Host language programmes, on the other hand, work in a row at a time. Cursors are used to move through a set of rows produced by a SQL SELECT statement included in the code. A cursor is similar to a pointer.

What is the purpose of normalization in DBMS?
The practise of structuring the attributes of a database in order to decrease or remove data redundancy is known as database normalisation (having the same data but at different places).

Normalization's purpose:

It is used to clean up the relational table by removing duplicate data and database oddities.

By assessing new data types utilised in the table, normalisation helps to decrease redundancy and complexity.

It's a good idea to break down a huge database table into smaller tables and use relationships to connect them.

It prevents duplicate data from being entered into a database, as well as no recurring groups.

It lowers the likelihood of anomalies in a database.

What is the difference between a database schema and a database state?
Database state refers to the collection of data kept in a database at a specific point in time, whereas database schema refers to the database's overall design.

What is the purpose of SQL?
SQL stands for Structured Query Language, and its primary purpose is to interact with relational databases by entering, updating, and altering data in the database.

Explain the concepts of a Primary key and Foreign Key.
Primary Key is used to uniquely identify records in a database table, whereas Foreign Key is used to connect two or more tables together, since it is a specific field(s) in one database table that is the primary key of another table.
Employee and Department are two tables, for example. Both tables have a similar field/column called 'ID,' which is the primary key for the Employee table and the foreign key for the Department table.

What are the main differences between Primary key and Unique Key?
A few discrepancies are listed below:

The primary distinction between the Primary and Unique keys is that the Primary key can never contain a null value, whereas the Unique key can.
There can only be one main key in each table, although there can be multiple unique keys in a table.

What is the concept of sub-query in terms of SQL?
A sub-query is a query that is contained within another query. It is also known as an inner query because it is found within the outer query.

What is the use of the DROP command and what are the differences between DROP, TRUNCATE and DELETE commands?
The DROP command is a DDL command that deletes an existing table, database, index, or view from a database.
The following are the main differences between the DROP, TRUNCATE, and DELETE commands:

The DDL commands DROP and TRUNCATE are used to delete tables from the database, and after the table is gone, all rights and indexes associated with the table are likewise deleted. Because these two actions cannot be reversed, they should only be utilised when absolutely required.
The DELETE command, on the other hand, is a DML command that may also be rolled back to delete rows from a table.

What is the main difference between UNION and UNION ALL?
UNION and UNION ALL are both used to connect data from two or more tables, but UNION removes duplicate rows and selects the distinct rows after merging the data from the tables, whilst UNION ALL does not remove duplicate rows and simply selects all of the data from the tables.

What is Correlated Subquery in DBMS?
A nested query, or a query written inside another query, is known as a subquery. A Correlated Subquery is defined as a Subquery that is conducted for each row of the outer query.

What integrity rules exist in the DBMS?
In a database management system, there are two major integrity rules.

🚀Entity Integrity: This is a crucial rule that stipulates that the value of a primary key can never be NULL.
🚀Referential Integrity: This rule is connected to the Foreign key and stipulates that a Foreign key's value must be NULL or it must be the primary key of another relation.

What is E-R model in the DBMS?
In relational databases, the E-R model is known as an Entity-Relationship model, and it is built on the concept of Entities and the relationships that exist between them.


################################################################
################################################################




Congratulations on getting an interview with Netflix for a Senior Data Engineer position! To help you prepare, I'll provide a detailed overview of ETL (Extract, Transform, Load), data modeling, and SQL. Let's break down each topic:

ETL (Extract, Transform, Load)
ETL is a process in data warehousing that involves extracting data from various sources, transforming it into a suitable format, and loading it into a data warehouse or other data storage system.

Extract:

Definition: Extracting data means retrieving data from various source systems.
Sources: Databases (SQL/NoSQL), APIs, flat files (CSV, JSON), and more.
Tools: Apache Nifi, Talend, Informatica, custom scripts (Python, Bash).
Transform:

Definition: Transforming data involves cleaning, aggregating, and modifying the data to fit the desired format or structure.
Operations: Data cleaning, filtering, deduplication, aggregation, joining, and data type conversion.
Tools: Apache Spark, AWS Glue, DBT (Data Build Tool), SQL, Pandas (Python).
Load:

Definition: Loading data involves writing the transformed data into a target data storage system.
Targets: Data warehouses (Snowflake, Redshift, BigQuery), databases (PostgreSQL, MySQL), and data lakes (S3, HDFS).
Tools: Apache Airflow, AWS Data Pipeline, custom scripts (Python, SQL).
Example ETL Workflow:
Extract data from an S3 bucket and a PostgreSQL database.
Transform the data using Pandas to clean and aggregate it.
Load the transformed data into a Redshift data warehouse.
Data Modeling
Data modeling is the process of designing the structure of a database or data warehouse. It involves defining tables, columns, relationships, and constraints.

Conceptual Data Model:

High-Level: Represents the overall structure of the database without going into details.
Entities and Relationships: Focuses on what data is stored and how entities are related.
Logical Data Model:

Details: More detailed than the conceptual model, specifying the data structures (tables, columns, data types).
Normalization: Ensures minimal redundancy and optimal organization of data.
Physical Data Model:

Implementation: Specifies how the logical data model will be implemented in a specific database system.
Indexes, Partitions: Includes details about indexes, partitions, and storage details.
Example Data Model:
Entities: Customers, Orders, Products.
Relationships: One customer can place multiple orders; each order can include multiple products.
SQL (Structured Query Language)
SQL is a standard language for managing and manipulating databases. Here are some key concepts and commands:

Basic SQL Commands:

SELECT: Retrieve data from one or more tables.
sql
Copy code
SELECT column1, column2 FROM table_name;
INSERT: Add new records to a table.
sql
Copy code
INSERT INTO table_name (column1, column2) VALUES (value1, value2);
UPDATE: Modify existing records.
sql
Copy code
UPDATE table_name SET column1 = value1 WHERE condition;
DELETE: Remove records from a table.
sql
Copy code
DELETE FROM table_name WHERE condition;
Advanced SQL Concepts:

JOIN: Combine rows from two or more tables.
sql
Copy code
SELECT a.column1, b.column2
FROM table1 a
JOIN table2 b ON a.common_field = b.common_field;
GROUP BY: Group rows that have the same values in specified columns.
sql
Copy code
SELECT column1, COUNT(*)
FROM table_name
GROUP BY column1;
HAVING: Filter groups after the GROUP BY clause.
sql
Copy code
SELECT column1, COUNT(*)
FROM table_name
GROUP BY column1
HAVING COUNT(*) > 1;
Subqueries: Nested queries within a query.
sql
Copy code
SELECT column1
FROM table_name
WHERE column2 IN (SELECT column2 FROM another_table WHERE condition);
Practical Example
Let's create a small ETL pipeline using SQL and Python for a hypothetical scenario:

Extract data from two sources: a CSV file and a PostgreSQL database.
Transform the data using Pandas.
Load the data into a PostgreSQL database.
Extract
python
Copy code
import pandas as pd
import psycopg2

# Extract from CSV
csv_data = pd.read_csv('data.csv')

# Extract from PostgreSQL
conn = psycopg2.connect("dbname=test user=postgres password=secret")
sql_query = "SELECT * FROM source_table"
pg_data = pd.read_sql_query(sql_query, conn)
Transform
python
Copy code
# Clean and transform the data
csv_data['column1'] = csv_data['column1'].str.lower()
pg_data['column2'] = pg_data['column2'].fillna(0)

# Merge data
merged_data = pd.merge(csv_data, pg_data, on='common_column')
Load
python
Copy code
# Load data into PostgreSQL
merged_data.to_sql('target_table', conn, if_exists='replace', index=False)
Final Tips
Brush up on SQL: Practice writing complex queries, including joins, subqueries, and window functions.
Understand Data Modeling: Be prepared to design a data model based on a business scenario.
ETL Tools: Familiarize yourself with at least one ETL tool or framework.
Good luck with your interview! If you have any specific questions or need further assistance, feel free to ask.

teach me more data modeling + sql tuning stuff
ChatGPT
Data Modeling
Data modeling is crucial for designing efficient and scalable databases. Let's delve into more advanced topics such as normalization, denormalization, and dimensional modeling.

Normalization
Normalization is the process of organizing data to reduce redundancy and improve data integrity. There are several normal forms (NFs), each with specific rules:

First Normal Form (1NF):

Eliminate duplicate columns from the same table.
Create separate tables for each group of related data.
Identify each row with a unique column or set of columns (primary key).
Second Normal Form (2NF):

Meet all the requirements of the first normal form.
Remove subsets of data that apply to multiple rows of a table and place them in separate tables.
Create relationships between these new tables and their predecessors with foreign keys.
Third Normal Form (3NF):

Meet all the requirements of the second normal form.
Remove columns that are not dependent upon the primary key.
Denormalization
Denormalization is the process of combining tables to optimize read performance. It can involve adding redundant data to reduce the number of joins:

Pros: Improved read performance, simplified queries.
Cons: Increased storage requirements, potential for data anomalies.
Dimensional Modeling
Dimensional modeling is often used in data warehousing and involves structuring data into fact and dimension tables:

Fact Tables:

Store quantitative data for analysis.
Typically include measures (e.g., sales amount) and foreign keys to dimension tables.
Dimension Tables:

Store descriptive attributes related to the facts (e.g., customer name, product category).
Support filtering, grouping, and labeling.
Star Schema
A star schema is a type of dimensional model with a central fact table connected to multiple dimension tables.



Snowflake Schema
A snowflake schema is a more normalized form of a star schema, where dimension tables are further normalized into multiple related tables.



SQL Tuning
SQL tuning involves optimizing SQL queries to improve performance. Here are some key concepts and techniques:

Indexing
Indexes speed up data retrieval but can slow down write operations. Choose indexes based on query patterns:

B-Tree Indexes: Good for most use cases, support range queries.
Hash Indexes: Efficient for equality comparisons but not range queries.
Bitmap Indexes: Suitable for columns with a low cardinality (few unique values).
Example:

sql
Copy code
CREATE INDEX idx_customer_name ON customers (name);
Query Optimization
EXPLAIN Plan:

Use EXPLAIN to understand how a query is executed.
sql
Copy code
EXPLAIN SELECT * FROM orders WHERE customer_id = 1;
**Avoid SELECT ***:

Select only necessary columns to reduce data transfer.
sql
Copy code
SELECT id, name FROM customers WHERE active = true;
Proper Use of JOINS:

Use INNER JOIN for filtering data.
Use LEFT JOIN for including all rows from the left table.
Example:

sql
Copy code
SELECT o.order_id, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id
WHERE c.active = true;
Use of WHERE Clause:

Filter data early in the query to reduce the dataset size.
sql
Copy code
SELECT * FROM orders WHERE order_date > '2023-01-01';
Limit the Result Set:

Use LIMIT to restrict the number of rows returned.
sql
Copy code
SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;
Caching
Database Caching:
Use caching mechanisms like Redis or Memcached to store frequently accessed data.
Query Result Caching:
Cache results of expensive queries to reduce database load.
Partitioning
Partitioning divides a large table into smaller, more manageable pieces without physically separating the data:

Horizontal Partitioning:

Split rows into different tables based on a column value (e.g., date range).
sql
Copy code
CREATE TABLE orders_2023 PARTITION OF orders FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
Vertical Partitioning:

Split columns into different tables, useful for frequently accessed columns.
sql
Copy code
CREATE TABLE customer_contact AS SELECT id, phone, email FROM customers;
Practice Scenario
Let's combine the concepts with a practical example. Consider an e-commerce database with orders, customers, and products:

Data Model:

Fact Table: orders
Dimension Tables: customers, products
SQL Queries:

Create Tables:

sql
Copy code
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    active BOOLEAN
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    product_id INT REFERENCES products(id),
    order_date DATE,
    quantity INT
);
Indexes:

sql
Copy code
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_orders_product_id ON orders (product_id);
Query Optimization:

sql
Copy code
EXPLAIN ANALYZE
SELECT o.id, c.name, p.name, o.quantity
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id
INNER JOIN products p ON o.product_id = p.id
WHERE o.order_date > '2023-01-01';
Partitioning Example:

sql
Copy code
CREATE TABLE orders_2023 PARTITION OF orders
FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
By understanding these advanced data modeling techniques and SQL tuning strategies, you'll be well-prepared for your interview and capable of designing efficient and scalable data systems. If you have any specific questions or need further clarification, feel free to ask!

teach me more for interview
ChatGPT
Certainly! Here are more advanced topics and techniques to help you prepare for your interview:

Advanced Data Modeling
Slowly Changing Dimensions (SCD)
In data warehousing, Slowly Changing Dimensions are used to manage and track changes in dimension data over time.

Type 1 (Overwrite):

Update the existing record with new data.
No historical data is preserved.
sql
Copy code
UPDATE customer_dimension
SET name = 'John Doe', address = 'New Address'
WHERE customer_id = 1;
Type 2 (Versioning):

Add a new row for each change, preserving historical data.
Use start and end date columns to track changes.
sql
Copy code
INSERT INTO customer_dimension (customer_id, name, address, start_date, end_date)
VALUES (1, 'John Doe', 'New Address', '2024-01-01', NULL);

UPDATE customer_dimension
SET end_date = '2023-12-31'
WHERE customer_id = 1 AND end_date IS NULL;
Type 3 (Add New Attribute):

Add new columns to store previous values.
Limited historical data is preserved.
sql
Copy code
ALTER TABLE customer_dimension
ADD COLUMN previous_address VARCHAR(255);

UPDATE customer_dimension
SET previous_address = address, address = 'New Address'
WHERE customer_id = 1;
Fact Table Design
Granularity:

Define the level of detail in a fact table.
High granularity means more detailed data.
Example: A sales fact table might store data at the transaction level or daily summary level.
Additive, Semi-Additive, Non-Additive Facts:

Additive: Facts that can be summed across any dimension (e.g., sales amount).
Semi-Additive: Facts that can be summed across some dimensions but not all (e.g., account balance over time).
Non-Additive: Facts that cannot be summed (e.g., ratios, percentages).
Factless Fact Tables:

Capture events or many-to-many relationships without numerical measures.
Example: Tracking student course enrollments.
sql
Copy code
CREATE TABLE student_enrollments (
    student_id INT,
    course_id INT,
    enrollment_date DATE
);
Advanced SQL Techniques
Window Functions
Window functions perform calculations across a set of table rows related to the current row.

ROW_NUMBER:

Assigns a unique number to each row within a partition.
sql
Copy code
SELECT
    customer_id,
    order_date,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS row_num
FROM orders;
RANK:

Assigns a rank to each row within a partition, with gaps in the ranking.
sql
Copy code
SELECT
    customer_id,
    order_date,
    RANK() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rank
FROM orders;
LEAD and LAG:

Access data from the subsequent or preceding row in the same result set.
sql
Copy code
SELECT
    customer_id,
    order_date,
    LAG(order_date, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_order_date
FROM orders;
SUM, AVG, MIN, MAX:

Perform aggregate calculations over a window of rows.
sql
Copy code
SELECT
    customer_id,
    order_date,
    SUM(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS running_total
FROM orders;
Common Table Expressions (CTEs)
CTEs simplify complex queries by breaking them into simpler parts.

Basic CTE:

sql
Copy code
WITH customer_orders AS (
    SELECT customer_id, COUNT(*) AS order_count
    FROM orders
    GROUP BY customer_id
)
SELECT c.customer_id, c.name, co.order_count
FROM customers c
JOIN customer_orders co ON c.customer_id = co.customer_id;
Recursive CTE:

Useful for hierarchical data.
sql
Copy code
WITH RECURSIVE employee_hierarchy AS (
    SELECT employee_id, manager_id, employee_name, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.manager_id, e.employee_name, eh.level + 1
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy;
Query Optimization Techniques
Indexing Strategies
Composite Indexes:

Create indexes on multiple columns to optimize queries filtering by multiple criteria.
sql
Copy code
CREATE INDEX idx_customer_order ON orders (customer_id, order_date);
Covering Indexes:

Include all columns in the query to avoid accessing the table itself.
sql
Copy code
CREATE INDEX idx_order_details ON orders (customer_id, order_date, order_amount);
Unique Indexes:

Ensure column values are unique, improving query performance and data integrity.
sql
Copy code
CREATE UNIQUE INDEX idx_unique_email ON customers (email);
Query Hints
Force Index Use:

Direct the optimizer to use a specific index.
sql
Copy code
SELECT * FROM orders FORCE INDEX (idx_customer_order) WHERE customer_id = 1;
Optimize Join Order:

Use hints to specify the join order.
sql
Copy code
SELECT /*+ ORDERED */ * FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.order_date > '2023-01-01';
Analyzing Execution Plans
EXPLAIN Plan:

Analyze query execution plans to identify bottlenecks.
sql
Copy code
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 1 AND order_date > '2023-01-01';
Optimizing Execution Plans:

Identify slow operations like full table scans, sort operations, and nested loops.
Refactor queries, add indexes, or adjust database configuration based on insights from execution plans.
Preparing for the Interview
Study Common Data Engineering Problems:

Look into case studies or common data engineering challenges Netflix or similar companies might face.
Practice designing ETL pipelines and data models based on these scenarios.
Brush Up on Big Data Technologies:

Familiarize yourself with big data technologies like Apache Hadoop, Spark, Kafka, and cloud data platforms (AWS, Google Cloud, Azure).
Practical SQL Exercises:

Work on complex SQL problems on platforms like LeetCode, HackerRank, or SQLZoo to improve your query writing and optimization skills.
System Design:

Be prepared to discuss system design and architecture, including trade-offs, scalability, and performance considerations.
By mastering these advanced concepts and techniques, you'll be well-prepared to demonstrate your expertise and problem-solving skills during your interview. Good luck! If you have any specific questions or need further assistance, feel free to ask.

teach me me more
ChatGPT
Absolutely, let's dive deeper into advanced topics that can further bolster your preparation for the interview. Here we'll cover:

Advanced Data Modeling Techniques
Advanced SQL Performance Tuning
Big Data and Distributed Systems
ETL Best Practices and Tools
Advanced Data Modeling Techniques
1. Data Vault Modeling
Data Vault is a method for modeling the data warehouse that separates the raw data from business logic:

Hubs: Central tables containing unique business keys (e.g., customer ID).

sql
Copy code
CREATE TABLE HubCustomer (
    CustomerID INT PRIMARY KEY,
    LoadDate TIMESTAMP
);
Satellites: Descriptive data related to hubs, including historical changes.

sql
Copy code
CREATE TABLE SatCustomer (
    CustomerID INT,
    Name VARCHAR(100),
    Address VARCHAR(255),
    EffectiveDate TIMESTAMP,
    EndDate TIMESTAMP
);
Links: Relationships between hubs.

sql
Copy code
CREATE TABLE LinkOrder (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    ProductID INT,
    LoadDate TIMESTAMP,
    FOREIGN KEY (CustomerID) REFERENCES HubCustomer(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES HubProduct(ProductID)
);
2. Anchor Modeling
Anchor modeling focuses on flexibility and extensibility:

Anchor Tables: Store immutable identifiers (e.g., CustomerID).

sql
Copy code
CREATE TABLE CustomerAnchor (
    CustomerID INT PRIMARY KEY
);
Attribute Tables: Store properties with timestamps.

sql
Copy code
CREATE TABLE CustomerName (
    CustomerID INT,
    Name VARCHAR(100),
    ValidFrom TIMESTAMP,
    FOREIGN KEY (CustomerID) REFERENCES CustomerAnchor(CustomerID)
);
Tie Tables: Store relationships.

sql
Copy code
CREATE TABLE OrderTie (
    OrderID INT,
    CustomerID INT,
    ProductID INT,
    ValidFrom TIMESTAMP,
    FOREIGN KEY (CustomerID) REFERENCES CustomerAnchor(CustomerID),
    FOREIGN KEY (ProductID) REFERENCES ProductAnchor(ProductID)
);
Advanced SQL Performance Tuning
1. Query Rewriting
Rewrite queries to optimize performance:

Avoid Subqueries:

sql
Copy code
-- Inefficient
SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE active = TRUE);

-- Efficient
SELECT o.* FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.active = TRUE;
Using EXISTS:

sql
Copy code
-- Using IN
SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE active = TRUE);

-- Using EXISTS
SELECT o.* FROM orders o
WHERE EXISTS (SELECT 1 FROM customers c WHERE c.id = o.customer_id AND c.active = TRUE);
2. Advanced Indexing
Partial Indexes: Create indexes on a subset of data.

sql
Copy code
CREATE INDEX idx_active_customers ON customers (id) WHERE active = TRUE;
Covering Indexes: Include columns in the index to avoid accessing the table.

sql
Copy code
CREATE INDEX idx_order_details ON orders (customer_id, order_date, order_amount);
Function-Based Indexes: Index based on expressions.

sql
Copy code
CREATE INDEX idx_upper_name ON customers (UPPER(name));
3. Parallel Query Execution
Utilize database features for parallel execution:

Enable Parallel Execution:

sql
Copy code
SET max_parallel_workers_per_gather = 4;
Partitioning for Parallelism:

sql
Copy code
CREATE TABLE orders_2023 PARTITION OF orders
FOR VALUES FROM ('2023-01-01') TO ('2023-12-31');
Big Data and Distributed Systems
1. Distributed Data Processing
Apache Spark: Used for large-scale data processing.

python
Copy code
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Example").getOrCreate()
df = spark.read.csv("data.csv")
df.show()
Hadoop MapReduce: Process large datasets with a distributed algorithm.

java
Copy code
// Java example for Hadoop MapReduce
public class WordCount {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>{
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }
}
2. Data Storage Solutions
HDFS (Hadoop Distributed File System): Scalable storage.

bash
Copy code
hdfs dfs -put localfile.txt /user/hadoop/
NoSQL Databases: Optimized for specific use cases.

Cassandra: For high write throughput.
cql
Copy code
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    name TEXT,
    age INT
);
MongoDB: For document storage.
javascript
Copy code
db.users.insertOne({ "name": "John Doe", "age": 30 });
ETL Best Practices and Tools
1. ETL Best Practices
Data Quality: Ensure data accuracy, completeness, and consistency.
Error Handling: Implement robust error handling and logging.
Scalability: Design ETL pipelines to handle increasing data volumes.
Maintainability: Write modular, reusable code.
2. ETL Tools
Apache Airflow: Workflow management platform.

python
Copy code
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def print_hello():
    print("Hello world")

dag = DAG('hello_world', description='Simple tutorial DAG',
          schedule_interval='0 12 * * *',
          start_date=datetime(2023, 1, 1), catchup=False)

hello_operator = PythonOperator(task_id='hello_task', python_callable=print_hello, dag=dag)
AWS Glue: Fully managed ETL service.

python
Copy code
import boto3

glue = boto3.client('glue')

response = glue.create_job(
    Name='MyETLJob',
    Role='AWSGlueServiceRole',
    Command={
        'Name': 'glueetl',
        'ScriptLocation': 's3://my-bucket/scripts/my_script.py'
    }
)
dbt (Data Build Tool): Transform data in your warehouse.

sql
Copy code
-- models/my_model.sql
SELECT
    id,
    UPPER(name) AS name
FROM
    {{ ref('raw_customers') }}
Interview Tips
Understand the Company’s Data Infrastructure:

Research Netflix’s data stack and technologies they use (e.g., Spark, Kafka, AWS).
Prepare for System Design Questions:

Practice designing data systems and explaining your thought process.
Be ready to discuss trade-offs and scalability considerations.
Work on Communication Skills:

Clearly articulate your ideas and solutions.
Use diagrams to illustrate data flows and system architectures.
Mock Interviews:

Practice with peers or use platforms like Pramp or Interviewing.io for mock interviews.
By diving into these advanced topics and honing your skills, you'll be well-prepared for your Netflix interview. Good luck! If you have any specific areas you'd like to explore further, let me know!

teach me more 
ChatGPT
Sure, let's continue with more advanced topics and techniques to deepen your knowledge and prepare you for the Netflix interview. We'll cover:

Advanced Data Warehousing Concepts
Real-time Data Processing
Advanced SQL Techniques and Query Optimization
Data Engineering Design Patterns
Data Security and Governance
Advanced Data Warehousing Concepts
1. OLAP vs OLTP
OLTP (Online Transaction Processing):

Designed for transactional systems.
Optimized for fast writes and updates.
Example: Customer relationship management systems.
OLAP (Online Analytical Processing):

Designed for complex queries and analysis.
Optimized for read-heavy operations.
Example: Data warehouses and business intelligence tools.
2. Columnar Storage
Columnar Databases: Store data by columns rather than rows, which can significantly speed up read queries for analytical purposes.
Examples: Amazon Redshift, Google BigQuery, Apache Parquet.
sql
Copy code
-- Example of a query optimized for a columnar database
SELECT customer_id, SUM(order_amount)
FROM orders
GROUP BY customer_id;
3. Data Warehouse Automation
ETL Automation Tools:

Automate repetitive ETL tasks and data pipeline management.
Examples: Matillion, Apache Nifi, Talend.
Schema Evolution:

Handle changes in data schema over time without manual intervention.
Example: Tools like dbt (Data Build Tool) can help manage schema changes and transformations.
Real-time Data Processing
1. Stream Processing
Apache Kafka: Distributed streaming platform for building real-time data pipelines and streaming applications.

java
Copy code
// Java example for producing messages to Kafka
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

Producer<String, String> producer = new KafkaProducer<>(props);
producer.send(new ProducerRecord<>("my-topic", "key", "value"));
producer.close();
Apache Flink: Framework for stateful computations over data streams.

java
Copy code
// Example of Apache Flink for processing stream data
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
DataStream<String> text = env.socketTextStream("localhost", 9999);
DataStream<Tuple2<String, Integer>> wordCounts = text
    .flatMap(new LineSplitter())
    .keyBy(0)
    .sum(1);
wordCounts.print();
env.execute("Socket Window WordCount");
Apache Beam: Unified model for batch and stream data processing.

python
Copy code
import apache_beam as beam

def format_result(word, count):
    return f'{word}: {count}'

p = beam.Pipeline()
(p | 'Read' >> beam.io.ReadFromText('gs://my_bucket/input.txt')
   | 'CountWords' >> beam.FlatMap(lambda line: line.split())
   | 'PairWithOne' >> beam.Map(lambda word: (word, 1))
   | 'GroupAndSum' >> beam.CombinePerKey(sum)
   | 'Format' >> beam.MapTuple(format_result)
   | 'Write' >> beam.io.WriteToText('gs://my_bucket/output.txt'))
p.run().wait_until_finish()
Advanced SQL Techniques and Query Optimization
1. Advanced Window Functions
CUME_DIST: Calculates the cumulative distribution of a value in a set of values.

sql
Copy code
SELECT
    customer_id,
    order_amount,
    CUME_DIST() OVER (ORDER BY order_amount DESC) AS cume_dist
FROM orders;
NTILE: Divides rows into a specified number of groups and assigns a number to each group.

sql
Copy code
SELECT
    customer_id,
    order_amount,
    NTILE(4) OVER (ORDER BY order_amount DESC) AS quartile
FROM orders;
FIRST_VALUE and LAST_VALUE: Return the first and last value in a set of rows.

sql
Copy code
SELECT
    customer_id,
    order_date,
    FIRST_VALUE(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS first_order_amount,
    LAST_VALUE(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_order_amount
FROM orders;
2. Query Hints and Execution Plan Analysis
Optimizer Hints: Influence the SQL optimizer to choose a specific query execution plan.

sql
Copy code
SELECT /*+ INDEX(customers idx_customer_name) */ * FROM customers WHERE name = 'John Doe';
EXPLAIN and ANALYZE:

sql
Copy code
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 1;
Execution Plan Analysis:

Look for costly operations like full table scans, nested loops, and sorts.
Identify areas for adding indexes, partitioning, or query rewriting.
Data Engineering Design Patterns
1. Lambda Architecture
Combines batch and real-time processing:

Batch Layer: Processes large volumes of data in batches for accuracy.
Speed Layer: Processes real-time data streams for low-latency updates.
Serving Layer: Merges batch and real-time views for querying.


2. Kappa Architecture
Simplifies the Lambda architecture by using a single stream processing layer:

Stream Processing: All data is processed as streams.
Reprocessing: Reprocesses historical data by replaying the stream.


Data Security and Governance
1. Data Security
Encryption: Use encryption for data at rest and in transit.

Example: Enable SSL/TLS for database connections.
sql
Copy code
-- MySQL example
GRANT USAGE ON *.* TO 'user'@'host' REQUIRE SSL;
Access Control: Implement role-based access control (RBAC) and least privilege principles.

sql
Copy code
-- PostgreSQL example
CREATE ROLE analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;
Auditing: Enable database auditing to track access and changes to sensitive data.

sql
Copy code
-- Oracle example
AUDIT SELECT ON hr.employees BY ACCESS;
2. Data Governance
Data Lineage: Track data flow from source to destination.

Example tools: Apache Atlas, Collibra.
java
Copy code
// Example using Apache Atlas
AtlasClient atlasClient = new AtlasClient("http://localhost:21000/", "admin", "admin");
atlasClient.createEntity(new AtlasEntity("customer", "data_source", "MySQL", "sales_db"));
Data Quality: Implement data quality checks and monitoring.

Example tools: Great Expectations, Talend.
python
Copy code
from great_expectations.dataset import PandasDataset

df = PandasDataset(your_dataframe)
df.expect_column_values_to_not_be_null('customer_id')
df.expect_column_values_to_be_unique('order_id')
Master Data Management (MDM): Ensure consistency and accuracy of key business data.

Example tools: Informatica MDM, IBM InfoSphere MDM.
xml
Copy code
<!-- Example using IBM InfoSphere MDM -->
<DataModel>
    <Entity name="Customer" primaryKey="CustomerID">
        <Attribute name="Name" type="String"/>
        <Attribute name="Email" type="String"/>
    </Entity>
</DataModel>
Interview Tips
Scenario-Based Questions:

Be prepared to discuss how you would handle specific data engineering challenges.
Example: "How would you design a system to handle streaming data from IoT devices?"
Hands-On Tasks:

Expect to write SQL queries, design schemas, or solve data processing problems on the spot.
Practice coding in an environment similar to the interview setup (e.g., online coding platforms).
Problem-Solving Approach:

Demonstrate your problem-solving process: clarify requirements, discuss trade-offs, and justify your design choices.
Example: Explain why you chose a particular indexing strategy or partitioning scheme.
Stay Current:

Keep up-to-date with the latest trends and technologies in data engineering.
Read blogs, follow industry leaders on social media, and participate in relevant forums or communities.
By mastering these advanced topics and refining your skills, you'll be well-prepared to tackle even the most challenging interview questions and demonstrate your expertise. If you have any specific areas you'd like to delve deeper into, feel free to ask!

Explain these deeply with examples:

Database Normalization:

41. What is database normalization?
42. What are the main objectives of normalization?
43. What is the First Normal Form (1NF)?
44. What is the Second Normal Form (2NF)?
45. What is the Third Normal Form (3NF)?
46. What is the Boyce-Codd Normal Form (BCNF)?
47. What is the Fourth Normal Form (4NF)?
48. What is the Fifth Normal Form (5NF)?
49. What is denormalization, and when is it used?
50. What are the advantages and disadvantages of normalization?
ChatGPT
Sure, let's go through each of these concepts in detail with examples.

Database Normalization
41. What is database normalization?
Database normalization is the process of structuring a relational database in accordance with a series of normal forms in order to reduce data redundancy and improve data integrity. It involves dividing large tables into smaller, related tables and defining relationships between them.

42. What are the main objectives of normalization?
The main objectives of normalization are:

Eliminate Redundant Data: Reduce duplication of data.
Ensure Data Integrity and Consistency: Prevent anomalies during data operations.
Simplify the Structure: Make the database structure more flexible and easier to maintain.
Improve Query Performance: Optimize the performance of queries by organizing data efficiently.
43. What is the First Normal Form (1NF)?
1NF states that the values in each column of a table must be atomic (indivisible). Each column should contain only one value per row.

Example:

Before 1NF:

OrderID	CustomerName	Items
1	John Doe	Apple, Banana
2	Jane Smith	Orange, Mango, Grape
After 1NF:

OrderID	CustomerName	Item
1	John Doe	Apple
1	John Doe	Banana
2	Jane Smith	Orange
2	Jane Smith	Mango
2	Jane Smith	Grape
44. What is the Second Normal Form (2NF)?
2NF states that a table must be in 1NF, and all non-key attributes must be fully functional dependent on the primary key. This means that there should be no partial dependency of any column on the primary key.

Example:

Before 2NF:

OrderID	CustomerName	Item	ItemPrice
1	John Doe	Apple	1.00
1	John Doe	Banana	0.50
2	Jane Smith	Orange	0.75
After 2NF:

Orders Table:

OrderID	CustomerName
1	John Doe
2	Jane Smith
OrderDetails Table:

OrderID	Item	ItemPrice
1	Apple	1.00
1	Banana	0.50
2	Orange	0.75
45. What is the Third Normal Form (3NF)?
3NF states that a table must be in 2NF and all the attributes must be directly dependent on the primary key. There should be no transitive dependency.

Example:

Before 3NF:

OrderID	CustomerID	CustomerName	Item	ItemPrice	CustomerAddress
1	101	John Doe	Apple	1.00	Address 1
2	102	Jane Smith	Orange	0.75	Address 2
After 3NF:

Orders Table:

OrderID	CustomerID
1	101
2	102
Customers Table:

CustomerID	CustomerName	CustomerAddress
101	John Doe	Address 1
102	Jane Smith	Address 2
OrderDetails Table:

OrderID	Item	ItemPrice
1	Apple	1.00
2	Orange	0.75
46. What is the Boyce-Codd Normal Form (BCNF)?
BCNF is an extension of the Third Normal Form (3NF). A table is in BCNF if it is in 3NF and for every one of its non-trivial functional dependencies X -> Y, X is a super key.

Example:

Before BCNF:

StudentID	CourseID	Instructor
1	101	Dr. Smith
2	101	Dr. Smith
3	102	Dr. Jones
Here, Instructor depends on CourseID, which is not a superkey.

After BCNF:

Courses Table:

CourseID	Instructor
101	Dr. Smith
102	Dr. Jones
StudentCourses Table:

StudentID	CourseID
1	101
2	101
3	102
47. What is the Fourth Normal Form (4NF)?
4NF states that a table must be in BCNF and should not have any multi-valued dependencies. A multi-valued dependency occurs when one attribute in a table uniquely determines another attribute, regardless of the other attributes.

Example:

Before 4NF:

StudentID	Course	Activity
1	Math	Basketball
1	Math	Swimming
1	Science	Basketball
1	Science	Swimming
After 4NF:

StudentsCourses Table:

StudentID	Course
1	Math
1	Science
StudentsActivities Table:

StudentID	Activity
1	Basketball
1	Swimming
48. What is the Fifth Normal Form (5NF)?
5NF states that a table must be in 4NF and should not have any join dependency. A join dependency exists when a table can be decomposed into smaller tables and then joined back together without losing any information.

Example:

Before 5NF:

StudentID	Course	Instructor
1	Math	Dr. Smith
1	Science	Dr. Jones
2	Math	Dr. Smith
After 5NF:

StudentsCourses Table:

StudentID	Course
1	Math
1	Science
2	Math
CoursesInstructors Table:

Course	Instructor
Math	Dr. Smith
Science	Dr. Jones
StudentsInstructors Table:

StudentID	Instructor
1	Dr. Smith
1	Dr. Jones
2	Dr. Smith


49. What is denormalization, and when is it used?
Denormalization is the process of combining normalized tables into larger tables to improve query performance at the cost of data redundancy. It is used when read performance is critical, and the complexity of joins in normalized tables negatively impacts performance.

Example:

Before Denormalization (Normalized):

Orders Table
OrderID
Customers Table
CustomerID
After Denormalization:

OrderID	CustomerID	CustomerName	CustomerAddress
1	101	John Doe	Address 1
2	102	Jane Smith	Address 2

50. What are the advantages and disadvantages of normalization?
Advantages:

Reduced Data Redundancy: Minimizes duplication of data.
Improved Data Integrity: Ensures consistency and accuracy of data.
Easier Maintenance: Simplifies updates, insertions, and deletions.
Better Performance: Efficient use of storage space and improved performance for certain types of queries.
Disadvantages:

Complex Queries: Can lead to complex joins and queries.
Decreased Performance: May reduce performance for read-heavy operations due to the need for joins.
Development Overhead: Requires careful planning and design.
Inflexibility: Can make the schema less flexible for changes and expansions.
By understanding these concepts and their applications, you'll be well-prepared to discuss database normalization and related topics during your interview. If you have any specific questions or need further elaboration on any topic, feel free to ask!