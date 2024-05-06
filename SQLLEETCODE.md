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
