1555)1555. Bank Account Summary
        Solved
        Medium
        Topics
        Companies
        SQL Schema
        Pandas Schema
        Table: Users

        +--------------+---------+
        | Column Name  | Type    |
        +--------------+---------+
        | user_id      | int     |
        | user_name    | varchar |
        | credit       | int     |
        +--------------+---------+
        user_id is the primary key (column with unique values) for this table.
        Each row of this table contains the current credit information for each user.
        

        Table: Transactions

        +---------------+---------+
        | Column Name   | Type    |
        +---------------+---------+
        | trans_id      | int     |
        | paid_by       | int     |
        | paid_to       | int     |
        | amount        | int     |
        | transacted_on | date    |
        +---------------+---------+
        trans_id is the primary key (column with unique values) for this table.
        Each row of this table contains information about the transaction in the bank.
        User with id (paid_by) transfer money to user with id (paid_to).
        

        Leetcode Bank (LCB) helps its coders in making virtual payments. Our bank records all transactions in the table Transaction, we want to find out the current balance of all users and check whether they have breached their credit limit (If their current credit is less than 0).

        Write a solution to report.

        user_id,
        user_name,
        credit, current balance after performing transactions, and
        credit_limit_breached, check credit_limit ("Yes" or "No")
        Return the result table in any order.

        The result format is in the following example.

        

        Example 1:

        Input: 
        Users table:
        +------------+--------------+-------------+
        | user_id    | user_name    | credit      |
        +------------+--------------+-------------+
        | 1          | Moustafa     | 100         |
        | 2          | Jonathan     | 200         |
        | 3          | Winston      | 10000       |
        | 4          | Luis         | 800         | 
        +------------+--------------+-------------+
        Transactions table:
        +------------+------------+------------+----------+---------------+
        | trans_id   | paid_by    | paid_to    | amount   | transacted_on |
        +------------+------------+------------+----------+---------------+
        | 1          | 1          | 3          | 400      | 2020-08-01    |
        | 2          | 3          | 2          | 500      | 2020-08-02    |
        | 3          | 2          | 1          | 200      | 2020-08-03    |
        +------------+------------+------------+----------+---------------+
        Output: 
        +------------+------------+------------+-----------------------+
        | user_id    | user_name  | credit     | credit_limit_breached |
        +------------+------------+------------+-----------------------+
        | 1          | Moustafa   | -100       | Yes                   | 
        | 2          | Jonathan   | 500        | No                    |
        | 3          | Winston    | 9900       | No                    |
        | 4          | Luis       | 800        | No                    |
        +------------+------------+------------+-----------------------+
        Explanation: 
        Moustafa paid $400 on "2020-08-01" and received $200 on "2020-08-03", credit (100 -400 +200) = -$100
        Jonathan received $500 on "2020-08-02" and paid $200 on "2020-08-08", credit (200 +500 -200) = $500
        Winston received $400 on "2020-08-01" and paid $500 on "2020-08-03", credit (10000 +400 -500) = $9990
        Luis did not received any transfer, credit = $800


            -- Write your PostgreSQL query statement below
            WITH loss as (SELECT u.user_id, SUM(t.amount) amt
            FROM users u 
            JOIN transactions t on t.paid_by = u.user_id 
            GROUP BY u.user_id), 

                gain as ( SELECT user_id, SUM(t.amount) amt
                FROM users u 
                JOIN transactions t on t.paid_to = u.user_id 
                GROUP BY u.user_id
                )
            SELECT u.user_id, u.user_name, coalesce(g.amt, 0) - coalesce(l.amt, 0) + u.credit as credit, 
            CASE
                WHEN coalesce(g.amt, 0) - coalesce(l.amt, 0) + u.credit < 0 THEN 'Yes'
            ELSE 'No'
            END as credit_limit_breached

            FROM users u 
            LEFT OUTER JOIN loss l ON l.user_id = u.user_id 
                LEFT JOIN gain g ON g.user_id = u.user_id;




2701. Consecutive Transactions with Increasing Amounts:
        Hard
        Topics
        SQL Schema
        Pandas Schema
        Table: Transactions

        +------------------+------+
        | Column Name      | Type |
        +------------------+------+
        | transaction_id   | int  |
        | customer_id      | int  |
        | transaction_date | date |
        | amount           | int  |
        +------------------+------+
        transaction_id is the primary key of this table. 
        Each row contains information about transactions that includes unique (customer_id, transaction_date) along with the corresponding customer_id and amount.  
        Write an SQL query to find the customers who have made consecutive transactions with increasing amount for at least three consecutive days. Include the customer_id, start date of the consecutive transactions period and the end date of the consecutive transactions period. There can be multiple consecutive transactions by a customer.

        Return the result table ordered by customer_id in ascending order.

        The query result format is in the following example.

        

        Example 1:

        Input: 
        Transactions table:
        +----------------+-------------+------------------+--------+
        | transaction_id | customer_id | transaction_date | amount |
        +----------------+-------------+------------------+--------+
        | 1              | 101         | 2023-05-01       | 100    |
        | 2              | 101         | 2023-05-02       | 150    |
        | 3              | 101         | 2023-05-03       | 200    |
        | 4              | 102         | 2023-05-01       | 50     |
        | 5              | 102         | 2023-05-03       | 100    |
        | 6              | 102         | 2023-05-04       | 200    |
        | 7              | 105         | 2023-05-01       | 100    |
        | 8              | 105         | 2023-05-02       | 150    |
        | 9              | 105         | 2023-05-03       | 200    |
        | 10             | 105         | 2023-05-04       | 300    |
        | 11             | 105         | 2023-05-12       | 250    |
        | 12             | 105         | 2023-05-13       | 260    |
        | 13             | 105         | 2023-05-14       | 270    |
        +----------------+-------------+------------------+--------+
        Output: 
        +-------------+-------------------+-----------------+
        | customer_id | consecutive_start | consecutive_end | 
        +-------------+-------------------+-----------------+
        | 101         |  2023-05-01       | 2023-05-03      | 
        | 105         |  2023-05-01       | 2023-05-04      |
        | 105         |  2023-05-12       | 2023-05-14      | 
        +-------------+-------------------+-----------------+
        Explanation: 
        - customer_id 101 has made consecutive transactions with increasing amounts from May 1st, 2023, to May 3rd, 2023
        - customer_id 102 does not have any consecutive transactions for at least 3 days. 
        - customer_id 105 has two sets of consecutive transactions: from May 1st, 2023, to May 4th, 2023, and from May 12th, 2023, to May 14th, 2023. 
        customer_id is sorted in ascending order.



        Dataset 1
        Description:
        To minimize our dataset let's filter for Customers + Dates with at least 1 valid consecutive day!

        Query:

        SELECT 
            a.customer_id, 
            a.transaction_date 
        FROM 
            Transactions a, 
            Transactions b 
        WHERE 
            a.customer_id = b.customer_id 
            AND b.amount > a.amount 
            AND DATEDIFF(b.transaction_date, a.transaction_date) = 1
        Output:

        +-------------+------------------+
        | customer_id | transaction_date |
        +-------------+------------------+
        | 101	      | 2023-05-01       |
        | 101	      | 2023-05-02       |
        | 102	      | 2023-05-03       |
        | 105	      | 2023-05-01       |
        | 105	      | 2023-05-02       |
        | 105	      | 2023-05-03       |
        | 105	      | 2023-05-12       |
        | 105	      | 2023-05-13       |
        +-------------+------------------+
        Dataset 2
        Description:
        Now, let's expand those customers to get Row Numbers of all their transactions!

        Hint: In the next step you'll see why we do this...
        Query:

        SELECT 
        customer_id, 
        transaction_date,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_date) AS rn
        FROM dataset_1
        Output:

        +-------------+------------------+----+
        | customer_id | transaction_date | rn |
        +-------------+------------------+----+
        | 101	      | 2023-05-01       |  1 |
        | 101	      |	2023-05-02       |	2 |
        | 102	      |	2023-05-03       |	1 |
        | 105	      |	2023-05-01       |	1 |
        | 105	      |	2023-05-02       |	2 |
        | 105	      |	2023-05-03       |	3 |
        | 105	      |	2023-05-12       |	4 |
        | 105	      |	2023-05-13       |	5 |
        +-------------+------------------+----+
        Dataset 3
        Description:
        With this enriched dataset we can now collect those rows into groups based on their row numbers.

        This is the confusing part! But it's a very popular pattern!

        Essentially the dates can have gaps between them, so by subtracting the current date by the row number we can get a unique date for each group!!!

        Query:

        SELECT 
            customer_id, 
            transaction_date, 
            DATE_SUB(transaction_date, INTERVAL rn DAY) AS date_group
        FROM dataset_2
        Output:

        +-------------+------------------+------------+
        | customer_id |	transaction_date | date_group |
        +-------------+------------------+------------+
        | 101	      |	2023-05-01       | 2023-04-30 |
        | 101	      |	2023-05-02       | 2023-04-30 | 
        | 102	      |	2023-05-03       | 2023-05-02 |
        | 105	      |	2023-05-01       | 2023-04-30 |
        | 105	      |	2023-05-02       | 2023-04-30 |
        | 105	      |	2023-05-03       | 2023-04-30 |
        | 105	      |	2023-05-12       | 2023-05-08 |
        | 105	      |	2023-05-13       | 2023-05-08 |
        +-------------+------------------+------------+
        Dataset 4
        Description:
        Now we simply count the # rows in that group to determine their size.

        Note: We can't use MAX(transaction_date) here as the last date in every group is not included due to the DATEDIFF(b.transaction_date, a.transaction_date) = 1 line in dataset1.

        Query:

        SELECT 
            customer_id, 
            MIN(transaction_date) AS consecutive_start, 
            COUNT(*) AS cnt
        FROM dataset_3 
        GROUP BY customer_id, date_group
        Output:

        +-------------+-------------------+-----+
        | customer_id |	consecutive_start | cnt |
        +-------------+-------------------+-----+
        | 101         |	2023-05-01        |	2   |
        | 102         |	2023-05-03        |	1   |
        | 105         |	2023-05-01        |	3   |
        | 105         |	2023-05-12        |	2   |
        +-------------+-------------------+-----+
        Dataset 5
        Description:
        Congrats! You made it! Now all we have to do is prettify the output!

        Query:

        SELECT 
            customer_id, 
            consecutive_start,
            DATE_ADD(consecutive_start, INTERVAL cnt DAY) AS consecutive_end 
        FROM dataset_4 
        WHERE cnt > 1 
        ORDER BY customer_id
        Output:

        +-------------+-------------------+-----------------+
        | customer_id |	consecutive_start |	consecutive_end |
        +-------------+-------------------+-----------------+
        | 101	      | 2023-05-01        |	2023-05-03      |
        | 105	      |	2023-05-01        |	2023-05-04      |
        | 105	      |	2023-05-12        |	2023-05-14      |
        +-------------+-------------------+-----------------+

        # 1. To minimize our dataset filter for Customers + Dates with at least 1 valid consecutive day!
        WITH dataset_1 AS (
            SELECT 
                a.customer_id, 
                a.transaction_date 
            FROM 
                Transactions a, 
                Transactions b 
            WHERE 
                a.customer_id = b.customer_id 
                AND b.amount > a.amount 
                AND DATEDIFF(b.transaction_date, a.transaction_date) = 1
        ),

        # 2. Expand those customers to get Row Numbers of all their transactions!
        dataset_2 AS (
            SELECT 
                customer_id, 
                transaction_date,
                ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_date) AS rn
            FROM dataset_1
        ),

        # 3. Collect those rows into groups based on their row numbers
        dataset_3 AS (
            SELECT 
                customer_id, 
                transaction_date, 
                DATE_SUB(transaction_date, INTERVAL rn DAY) AS date_group
            FROM dataset_2
        ),

        # 4. Count the # rows in that group to determine size
        dataset_4 AS (
            SELECT 
                customer_id, 
                MIN(transaction_date) AS consecutive_start, 
                COUNT(*) AS cnt
            FROM dataset_3 
            GROUP BY customer_id, date_group
        )

        # 5. Prettify the output!
        SELECT 
            customer_id, 
            consecutive_start,
            DATE_ADD(consecutive_start, INTERVAL cnt DAY) AS consecutive_end 
        FROM dataset_4 
        WHERE cnt > 1 
        ORDER BY customer_id 



570. Managers with at least 5 direct reports

        Table: Employee

        +-------------+---------+
        | Column Name | Type    |
        +-------------+---------+
        | id          | int     |
        | name        | varchar |
        | department  | varchar |
        | managerId   | int     |
        +-------------+---------+
        id is the primary key (column with unique values) for this table.
        Each row of this table indicates the name of an employee, their department, and the id of their manager.
        If managerId is null, then the employee does not have a manager.
        No employee will be the manager of themself.
        

        Write a solution to find managers with at least five direct reports.

        Return the result table in any order.

        The result format is in the following example.


        SELECT man.name
        from employee man, employee emp
        where emp.managerId = man.id  -- and emp.managerId = man.managerId 
        group by emp.managerId, man.name
        having count(emp.id) >= 5


180)    180. Consecutive Numbers

            +-------------+---------+
            | Column Name | Type    |
            +-------------+---------+
            | id          | int     |
            | num         | varchar |
            +-------------+---------+
            In SQL, id is the primary key for this table.
            id is an autoincrement column.
            

            Find all numbers that appear at least three times consecutively.

            Return the result table in any order.

            The result format is in the following example.

            

            Example 1:

            Input: 
            Logs table:
            +----+-----+
            | id | num |
            +----+-----+
            | 1  | 1   |
            | 2  | 1   |
            | 3  | 1   |
            | 4  | 2   |
            | 5  | 1   |
            | 6  | 2   |
            | 7  | 2   |
            +----+-----+
            Output: 
            +-----------------+
            | ConsecutiveNums |
            +-----------------+
            | 1               |
            +-----------------+
            Explanation: 1 is the only number that appears consecutively for at least three times.

            SELECT distinct a.num as ConsecutiveNums
            from Logs a, Logs b, Logs c
            where a.num = b.num and b.num = c.num 
            and a.num = c.num and a.id = b.id + 1 and b.id = c.id + 1

            

            Window function soln:

            SELECT DISTINCT num as ConsecutiveNums
            FROM
            (
            SELECT num,LEAD(num) OVER(ORDER BY id) AS lead, LAG(num) OVER (ORDER BY id) AS lag
            FROM Logs
            )t
            WHERE num=lead and num=lag


626) Exchange Seats:

    Table: Seat

        +-------------+---------+
        | Column Name | Type    |
        +-------------+---------+
        | id          | int     |
        | student     | varchar |
        +-------------+---------+
        id is the primary key (unique value) column for this table.
        Each row of this table indicates the name and the ID of a student.
        id is a continuous increment.
        

        Write a solution to swap the seat id of every two consecutive students. If the number of students is odd, the id of the last student is not swapped.

        Return the result table ordered by id in ascending order.

        The result format is in the following example.

        

        Example 1:

        Input: 
        Seat table:
        +----+---------+
        | id | student |
        +----+---------+
        | 1  | Abbot   |
        | 2  | Doris   |
        | 3  | Emerson |
        | 4  | Green   |
        | 5  | Jeames  |
        +----+---------+
        Output: 
        +----+---------+
        | id | student |
        +----+---------+
        | 1  | Doris   |
        | 2  | Abbot   |
        | 3  | Green   |
        | 4  | Emerson |
        | 5  | Jeames  |
        +----+---------+
        Explanation: 
        Note that if the number of students is odd, there is no need to change the last one's seat.

        Approach I: Using flow control statement CASE [Accepted]
        Algorithm

        For students with odd id, the new id is (id+1) after switch unless it is the last seat. 
        And for students with even id, the new id is (id-1). In order to know how many seats in total, we can use a subquery:

        SELECT
            (CASE
                WHEN MOD(id, 2) != 0 AND counts != id THEN id + 1
                WHEN MOD(id, 2) != 0 AND counts = id THEN id
                ELSE id - 1
            END) AS id,
            student
        FROM
            seat,
            (SELECT
                COUNT(*) AS counts
            FROM
                seat) AS seat_counts
        ORDER BY id ASC;



178 DISTINCT RANKS:
        Table: Scores

        +-------------+---------+
        | Column Name | Type    |
        +-------------+---------+
        | id          | int     |
        | score       | decimal |
        +-------------+---------+
        id is the primary key (column with unique values) for this table.
        Each row of this table contains the score of a game. Score is a floating point value with two decimal places.
        

        Write a solution to find the rank of the scores. The ranking should be calculated according to the following rules:

        The scores should be ranked from the highest to the lowest.
        If there is a tie between two scores, both should have the same ranking.
        After a tie, the next ranking number should be the next consecutive integer value. In other words, there should be no holes between ranks.
        Return the result table ordered by score in descending order.

        The result format is in the following example.

        

        Example 1:

        Input: 
        Scores table:
        +----+-------+
        | id | score |
        +----+-------+
        | 1  | 3.50  |
        | 2  | 3.65  |
        | 3  | 4.00  |
        | 4  | 3.85  |
        | 5  | 4.00  |
        | 6  | 3.65  |
        +----+-------+
        Output: 
        +-------+------+
        | score | rank |
        +-------+------+
        | 4.00  | 1    |
        | 4.00  | 1    |
        | 3.85  | 2    |
        | 3.65  | 3    |
        | 3.65  | 3    |
        | 3.50  | 4    |
        +-------+------+

    WITH distinct_ranks AS (
    SELECT a.score, count(b.score) as rank
    FROM (SELECT distinct score from Scores) a 
    JOIN (SELECT distinct score from Scores) b on b.score >= a.score
        group by a.score     
    ) SELECT x.score, d.rank 
    from Scores x, distinct_ranks d
    WHERE  x.score = d.score
        ORDER BY rank



        SELECT
        S1.score,
        (
            SELECT
            COUNT(DISTINCT S2.score)
            FROM
            Scores S2
            WHERE
            S2.score >= S1.score
        ) AS 'rank'
        FROM
        Scores S1
        ORDER BY
        S1.score DESC;



1934) Confirmation Rate

        Table: Signups

        +----------------+----------+
        | Column Name    | Type     |
        +----------------+----------+
        | user_id        | int      |
        | time_stamp     | datetime |
        +----------------+----------+
        user_id is the column of unique values for this table.
        Each row contains information about the signup time for the user with ID user_id.
        

        Table: Confirmations

        +----------------+----------+
        | Column Name    | Type     |
        +----------------+----------+
        | user_id        | int      |
        | time_stamp     | datetime |
        | action         | ENUM     |
        +----------------+----------+
        (user_id, time_stamp) is the primary key (combination of columns with unique values) for this table.
        user_id is a foreign key (reference column) to the Signups table.
        action is an ENUM (category) of the type ('confirmed', 'timeout')
        Each row of this table indicates that the user with ID user_id requested a confirmation message at time_stamp and that confirmation message was either confirmed ('confirmed') or expired without confirming ('timeout').
        

        The confirmation rate of a user is the number of 'confirmed' messages divided by the total number of requested confirmation messages. The confirmation rate of a user that did not request any confirmation messages is 0. Round the confirmation rate to two decimal places.

        Write a solution to find the confirmation rate of each user.

        Return the result table in any order.

        The result format is in the following example.

        

        Example 1:

        Input: 
        Signups table:
        +---------+---------------------+
        | user_id | time_stamp          |
        +---------+---------------------+
        | 3       | 2020-03-21 10:16:13 |
        | 7       | 2020-01-04 13:57:59 |
        | 2       | 2020-07-29 23:09:44 |
        | 6       | 2020-12-09 10:39:37 |
        +---------+---------------------+
        Confirmations table:
        +---------+---------------------+-----------+
        | user_id | time_stamp          | action    |
        +---------+---------------------+-----------+
        | 3       | 2021-01-06 03:30:46 | timeout   |
        | 3       | 2021-07-14 14:00:00 | timeout   |
        | 7       | 2021-06-12 11:57:29 | confirmed |
        | 7       | 2021-06-13 12:58:28 | confirmed |
        | 7       | 2021-06-14 13:59:27 | confirmed |
        | 2       | 2021-01-22 00:00:00 | confirmed |
        | 2       | 2021-02-28 23:59:59 | timeout   |
        +---------+---------------------+-----------+
        Output: 
        +---------+-------------------+
        | user_id | confirmation_rate |
        +---------+-------------------+
        | 6       | 0.00              |
        | 3       | 0.00              |
        | 7       | 1.00              |
        | 2       | 0.50              |
        +---------+-------------------+
        Explanation: 
        User 6 did not request any confirmation messages. The confirmation rate is 0.
        User 3 made 2 requests and both timed out. The confirmation rate is 0.
        User 7 made 3 requests and all were confirmed. The confirmation rate is 1.
        User 2 made 2 requests where one was confirmed and the other timed out. The confirmation rate is 1 / 2 = 0.5.



        SELECT s.user_id, CASE WHEN Count(c.user_id) = 0 THEN 0 
                       ELSE ROUND(SUM(CASE 
                        WHEN c.action = 'confirmed' THEN 1 
                        ELSE 0 
                        END) * 1.0 / COUNT(c.user_id), 2 ) 
                   END as confirmation_rate
        FROM Signups s 
        left Join Confirmations c on c.user_id = s.user_id 
        group by s.user_id

        SOln2 
    # Write your MySQL query statement below
    select s.user_id, round(avg(if(c.action="confirmed",1,0)),2) as confirmation_rate
    from Signups as s left join Confirmations as c on s.user_id= c.user_id group by user_id;



185) Department top 3 salaries:

    Table: Employee

        +--------------+---------+
        | Column Name  | Type    |
        +--------------+---------+
        | id           | int     |
        | name         | varchar |
        | salary       | int     |
        | departmentId | int     |
        +--------------+---------+
        id is the primary key (column with unique values) for this table.
        departmentId is a foreign key (reference column) of the ID from the Department table.
        Each row of this table indicates the ID, name, and salary of an employee. It also contains the ID of their department.
        

        Table: Department

        +-------------+---------+
        | Column Name | Type    |
        +-------------+---------+
        | id          | int     |
        | name        | varchar |
        +-------------+---------+
        id is the primary key (column with unique values) for this table.
        Each row of this table indicates the ID of a department and its name.
        

        A company's executives are interested in seeing who earns the most money in each of the company's departments. A high earner in a department is an employee who has a salary in the top three unique salaries for that department.

        Write a solution to find the employees who are high earners in each of the departments.

        Return the result table in any order.

        The result format is in the following example.

        

        Example 1:

        Input: 
        Employee table:
        +----+-------+--------+--------------+
        | id | name  | salary | departmentId |
        +----+-------+--------+--------------+
        | 1  | Joe   | 85000  | 1            |
        | 2  | Henry | 80000  | 2            |
        | 3  | Sam   | 60000  | 2            |
        | 4  | Max   | 90000  | 1            |
        | 5  | Janet | 69000  | 1            |
        | 6  | Randy | 85000  | 1            |
        | 7  | Will  | 70000  | 1            |
        +----+-------+--------+--------------+
        Department table:
        +----+-------+
        | id | name  |
        +----+-------+
        | 1  | IT    |
        | 2  | Sales |
        +----+-------+
        Output: 
        +------------+----------+--------+
        | Department | Employee | Salary |
        +------------+----------+--------+
        | IT         | Max      | 90000  |
        | IT         | Joe      | 85000  |
        | IT         | Randy    | 85000  |
        | IT         | Will     | 70000  |
        | Sales      | Henry    | 80000  |
        | Sales      | Sam      | 60000  |
        +------------+----------+--------+
        Explanation: 
        In the IT department:
        - Max earns the highest unique salary
        - Both Randy and Joe earn the second-highest unique salary
        - Will earns the third-highest unique salary

        In the Sales department:
        - Henry earns the highest salary
        - Sam earns the second-highest salary
        - There is no third-highest salary as there are only two employees



        SELECT d.name AS 'Department', 
            e1.name AS 'Employee', 
            e1.salary AS 'Salary' 
        FROM Employee e1
        JOIN Department d
        ON e1.departmentId = d.id 
        WHERE
            3 > (SELECT COUNT(DISTINCT e2.salary)
                FROM Employee e2
                WHERE e2.salary > e1.salary AND e1.departmentId = e2.departmentId);

    In the correlated subquery, we select the number of salaries from the same table Employee. 
    To compare the salaries between the main query and the subquery, we make sure the department 
    is the same from both queries, but the salary from the subquery is always bigger than the salary from the main query.

    (
        SELECT COUNT(DISTINCT e2.salary)
        FROM Employee e2
        WHERE e2.salary > e1.salary AND e1.departmentId = e2.departmentId
    )
    Since we need to identify the top three high earners in the main query, and 
    the subquery always has larger salaries than the salaries from the main query, 
    the maximum count of the larger salaries in the subquery is two. We add this criteria as a filter to the main query.




1454) Table: Accounts

        +---------------+---------+
        | Column Name   | Type    |
        +---------------+---------+
        | id            | int     |
        | name          | varchar |
        +---------------+---------+
        id is the primary key (column with unique values) for this table.
        This table contains the account id and the user name of each account.
        

        Table: Logins

        +---------------+---------+
        | Column Name   | Type    |
        +---------------+---------+
        | id            | int     |
        | login_date    | date    |
        +---------------+---------+
        This table may contain duplicate rows.
        This table contains the account id of the user who logged in and the login date. A user may log in multiple times in the day.
        

        Active users are those who logged in to their accounts for five or more consecutive days.

        Write a solution to find the id and the name of active users.

        Return the result table ordered by id.

        The result format is in the following example.


        Example 1:

        Input: 
        Accounts table:
        +----+----------+
        | id | name     |
        +----+----------+
        | 1  | Winston  |
        | 7  | Jonathan |
        +----+----------+
        Logins table:
        +----+------------+
        | id | login_date |
        +----+------------+
        | 7  | 2020-05-30 |
        | 1  | 2020-05-30 |
        | 7  | 2020-05-31 |
        | 7  | 2020-06-01 |
        | 7  | 2020-06-02 |
        | 7  | 2020-06-02 |
        | 7  | 2020-06-03 |
        | 1  | 2020-06-07 |
        | 7  | 2020-06-10 |
        +----+------------+
        Output: 
        +----+----------+
        | id | name     |
        +----+----------+
        | 7  | Jonathan |
        +----+----------+
        Explanation: 
        User Winston with id = 1 logged in 2 times only in 2 different days, so, Winston is not an active user.
        User Jonathan with id = 7 logged in 7 times in 6 different days, five of them were consecutive days, so, Jonathan is an active user.
        

        Follow up: Could you write a general solution if the active users are those who logged in to their accounts for n or more consecutive days?


        SELECT DISTINCT l1.id,
        (SELECT name FROM Accounts WHERE id = l1.id) AS name
        FROM Logins l1
        JOIN Logins l2 ON l1.id = l2.id AND (l2.login_date - l1.login_date) BETWEEN 1 AND 4
        GROUP BY l1.id, l1.login_date
        HAVING COUNT(DISTINCT l2.login_date) = 4

        Solution 2:

        SELECT *
        FROM Accounts
        WHERE id IN
            (SELECT DISTINCT t1.id 
            FROM Logins t1 INNER JOIN Logins t2 on t1.id = t2.id AND DATEDIFF(t1.login_date, t2.login_date) BETWEEN 1 AND 4
            GROUP BY t1.id, t1.login_date
            HAVING COUNT(DISTINCT(t2.login_date)) = 4)
        ORDER BY id


1285: Find the start and end number of contious ranges:

        Table: Logs

        +---------------+---------+
        | Column Name   | Type    |
        +---------------+---------+
        | log_id        | int     |
        +---------------+---------+
        log_id is the column of unique values for this table.
        Each row of this table contains the ID in a log Table.
        

        Write a solution to find the start and end number of continuous ranges in the table Logs.

        Return the result table ordered by start_id.

        The result format is in the following example.

        

        Example 1:

        Input: 
        Logs table:
        +------------+
        | log_id     |
        +------------+
        | 1          |
        | 2          |
        | 3          |
        | 7          |
        | 8          |
        | 10         |
        +------------+
        Output: 
        +------------+--------------+
        | start_id   | end_id       |
        +------------+--------------+
        | 1          | 3            |
        | 7          | 8            |
        | 10         | 10           |
        +------------+--------------+
        Explanation: 
        The result table should contain all ranges in table Logs.
        From 1 to 3 is contained in the table.
        From 4 to 6 is missing in the table
        From 7 to 8 is contained in the table.
        Number 9 is missing from the table.
        Number 10 is contained in the table.


        SELECT min(log_id) as start_id, max(log_id) as end_id
        FROM
        (SELECT log_id, ROW_NUMBER() OVER(ORDER BY log_id) as num
        FROM Logs) a
        GROUP BY log_id - num

        When you write down the following table, it becomes super easy to understand!

        log_id, num, difference
        1, 1, 0
        2, 2, 0
        3, 3, 0
        7, 4, 3
        8, 5, 3
        10, 6, 4