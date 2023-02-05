## Postgres

Types of constraints:  
 - Check  
 - Default  
 - Unique  
 - Not Null  

Primary key - uniqie inside the table  
Primary key may be composed from a set of values  

> Foreign keys can increase performance in reading data from multiple tables.
The query execution planner will have a better estimation of the number of rows that need to be processed.  

> Disabling foreign keys when doing a bulk insert will lead to a performance boost.  

Possible behaviors for foreign keys:  
 * Cascade: When a tuple is deleted or updated in the referenced relation, the tuples in the referencing relation are also updated or deleted  
 * Restrict: The tuple cannot be deleted or the referenced attribute cannot be updated if it is referenced by another relation  
 * No action: Similar to restrict, but it is deferred to the end of the transaction  
 * Set default: When a tuple in the referenced relation is deleted or the referenced attribute is updated, then the foreign key value is assigned the default value  
 * Set null: The foreign key attribute value is set to null when the referenced tuple is deleted  

#### 3 Common data model types:
 * Conceptual data model: Describes the domain semantics, and is used to communicate the main business rules, actors, and concepts. It describes the business requirements at a high level and is often called a high-level data model.  
 * Logical data model: Describes the semantics for a certain technology, for example, the UML class diagram for object-oriented languages.  
 * Physical data model: Describes how data is actually stored and manipulated at the hardware level such as storage area network, table space, CPUs, and so on.  

**Replication** is used for:  
 - High availability: A second server can take over if the primary server fails  
 - Load balancing: Several servers can serve the same requests  
 - Faster execution: A query is executed on several machines at once to gain performance  

Types of **Replication**:  
 - Streaming  
   - binary  
   - utilizes WAL-log  
   - data must be equal  
   - provides cascading replication (master -> slave -> slave)  
 - Logical  
   - translates WAL changes to logical changes
   - allows to create filters and keep part of data
   - allows to collect data from many master nodes

Also there are open source replication solutions:
 - *Slony-1* - This is a master to multiple slave replication systems
 - *Distributed Replicated Block Device (DRBD)* - a general solution for HA. It can be understood as a network RAID-1

#### PostgreSQL supports several authentication methods:  
 - trust  
 - password  
 - LDAB  
 - GSSAPI  
 - SSPI  
 - Kerberos  
 - ident-based  
 - RADUIS  
 - certificate  
 - PAM authentication  

PostgreSQL can use encryption to protect data by hardware encryption. Also, one can encrypt certain information by utilizing the pgcrypto extension.

#### Performance techniques
 - *Locking system* - PostgreSQL provides several types of locks at the table and row levels. PostgreSQL is able to use more granular locks that prevent locking/blocking more than necessary; this increases concurrency and decreases the blocking time  
 - *Indexes* - PostgreSQL provides six types of indexes: B-Tree, hash, generalized inverted index (GIN), and the Generalized Search Tree (GiST) index, SP- GiST, and Block Range Indexes (BRIN)  
 - Explain, analyze, vacuum, and cluster
 - Table inheritance and constraint exclusion
 - Very rich SQL constructs: correlated and uncorrelated subqueries, common table expressions, window functions, recursive queries.

#### Backup types:
 - physical - works faster, but requires a compatible version of postgres to restore data;  
 - logical - slower, dumps data in form of SQL-statements;  

#### PostgreSQL supports the next table types:
 - **Permanent table**: The table life span starts with table creation and ends with table dropping.  
 - **Temporary table**: The table life span is the user session. This is used often with procedural languages to model business logic.  
 - **Unlogged table**: Operations on unlogged tables are much faster than on permanent tables, because data is not written into the WAL files. Unlogged tables are not crash-safe. Also, since streaming replication is based on shipping the log files, unlogged tables cannot be replicated to the slave node.  
 - **Child table**: A child table is a table that inherits one or more tables. The inheritance is often used with constraint exclusion to physically partition the data on the hard disk and to improve performance by retrieving a subset of data that has a certain value.  

