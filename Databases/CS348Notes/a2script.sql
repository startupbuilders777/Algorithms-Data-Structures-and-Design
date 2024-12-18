-- A2 ANSWERS + SQL NOTES. SOME OF THESE ANSWERS MAY BE INCORRECT, VIEW OTHER A2 SHEET FOR CORRECT ANSWERS.
--  Schema definition for "Student-class-course" A2 (for DB2)
--
-- echo "droping preexisting tables"
/*
drop table department
drop table professor
drop table student
drop table course
drop table class
drop table schedule
drop table enrollment
drop table mark
*/
-- echo "creating tables"
/*
CREATE TABLE department (deptcode char(2) not null,  deptname varchar(20),   primary key (deptcode));

   
CREATE TABLE professor ( pnum     integer not null,  pname    varchar(30),  office   char(7),  deptcode char(2),  primary key (pnum),  foreign key (deptcode) references department(deptcode));

CREATE TABLE student (  snum      integer not null,  sname     varchar(30),  year      integer,   primary key (snum));

CREATE TABLE course ( deptcode  char(2) not null, cnum      integer not null, cname     varchar(50), primary key (deptcode,cnum), foreign key (deptcode) references department(deptcode));

CREATE TABLE class ( deptcode  char(2) not null,  cnum      integer not null, term      char(3) not null, section   integer not null, pnum      integer, primary key (deptcode,cnum,term,section),  foreign key (deptcode,cnum) references course(deptcode,cnum),  foreign key (pnum) references professor(pnum));

CREATE TABLE schedule ( deptcode  char(2) not null, cnum      integer not null, term      char(3) not null, section   integer not null,  day       varchar(10) not null, time      time not null,  room      char(7),  primary key (deptcode,cnum,term,section,day,time),  foreign key (deptcode,cnum,term,section) references class(deptcode,cnum,term,section));

CREATE TABLE enrollment ( snum      integer not null,  deptcode  char(2) not null, cnum      integer not null, term      char(3) not null, section   integer not null, primary key (snum,deptcode,cnum,term,section), foreign key (snum) references student(snum), foreign key (deptcode,cnum,term,section) references class(deptcode,cnum,term,section));

CREATE TABLE mark ( snum      integer not null, deptcode  char(2) not null,  cnum      integer not null,  term      char(3) not null,  section   integer not null,  grade     integer, primary key (snum,deptcode,cnum,term,section),  foreign key (snum,deptcode,cnum,term,section) references enrollment(snum,deptcode,cnum,term,section));






*/

/*
1. The student number and name of second year students who have obtained
a grade lower than 65 in at least two courses in a department with the
name “computer science”.
*/

/*

verbose solution:
select distinct student.snum as StudentNumber,  student.sname as Name 
from student, 
     department, 
     enrollment enrollment1, 
     enrollment enrollment2, 
     mark mark1, 
     mark mark2
where 	department.deptname = 'computer science' and
	department.deptcode = enrollment1.deptcode and
	department.deptcode = enrollment2.deptcode and 
	department.deptcode = mark1.deptcode and
	department.deptcode = mark2.deptcode and
	student.snum = enrollment1.snum  and
	student.snum = enrollment2.snum and
	student.snum = mark1.snum  and
	student.snum = mark2.snum and
	mark1.grade < 65 and
	mark2.grade < 65 and
	not (enrollment1.cnum = enrollment2.cnum and 
	enrollment1.term = enrollment2.term and
	enrollment1.section = enrollment2.section);
*/



select distinct student.snum as StudentNumber,  student.sname as Name 
from student, 
     department, 
     mark mark1, 
     mark mark2
where 	department.deptname = 'computer science' and
	department.deptcode = mark1.deptcode and
	department.deptcode = mark2.deptcode and 
	student.snum = mark1.snum  and
	student.snum = mark2.snum and
	mark1.grade < 65 and
	mark2.grade < 65 and
	not (mark1.cnum = mark2.cnum and 
	     mark1.term = mark2.term and
	     mark1.section = mark2.section);
	
/*

select distinct student.snum as StudentNumber,  student.sname as Name \
from student, \
     department, \ 
     mark mark1, \
     mark mark2 \
where 	department.deptname = 'computer science' and \
	department.deptcode = mark1.deptcode and \
	department.deptcode = mark2.deptcode and \
	student.snum = mark1.snum  and \
	student.snum = mark2.snum and \
	mark1.grade < 65 and \
	mark2.grade < 65 and \
	not (mark1.cnum = mark2.cnum and \ 
	     mark1.term = mark2.term and \
	     mark1.section = mark2.section)







*/

      



/*
2. The number and name of professors who are not in the pure math (PM)
department, and who are teaching CS245 for the first time. 
//so they are currently teaching CS245...
*/

-- You are teaching the class but you dont have a CS 245 class where marks are recorded for it.
select distinct professor.pnum as ProfessorNumber, professor.pname ProfessorName
from professor, class
where professor.deptcode <> 'PM' and 
      class.pnum = professor.pnum and
      class.deptcode = 'CS' and
      class.cnum = 245 and 
      not exists (select * 
		  from class class2, mark 	
		  where class2.pnum = professor.pnum and 
			class2.deptcode = 'CS' and 
			class2.cnum = 245 and
			mark.deptcode = 'CS' and 
			mark.cnum = 245 and
			mark.term = class2.term and 
			mark.section = class2.section);

/*

select distinct professor.pnum as ProfessorNumber, professor.pname ProfessorName \
from professor, class \
where professor.deptcode <> 'PM' and \ 
      class.pnum = professor.pnum and \
      class.deptcode = 'CS' and \
      class.cnum = 245 and \
      not exists (select * \
		  from class class2, mark \ 	
		  where class2.pnum = professor.pnum and \ 
			class2.deptcode = 'CS' and \
			class2.cnum = 245 and \
			mark.deptcode = 'CS' and \
			mark.cnum = 245 and \
			mark.term = class2.term and \
			mark.section = class2.section)




*/






/*
3. The number, name and year of each student who has obtained a grade in
CS240 that is within 3 marks of the highest ever grade recorded for that
course.
*/

select distinct s1.snum as StudentNumber, s1.sname as StudentName, s1.year as Year
from student s1, student smax, mark m1, mark mmax
where s1.snum = m1.snum and
      smax.snum = mmax.snum and
      m1.deptcode = 'CS' and
      m1.cnum = 240 and
      mmax.deptcode = 'CS' and
      mmax.cnum = 240 and
      not exists (select * from mark where mark.deptcode = 'CS' and mark.cnum = 240 and mmax.grade < mark.grade) and
      m1.grade >= mmax.grade - 3;   
      
      
/*

select distinct s1.snum as StudentNumber, s1.sname as StudentName, s1.year as Year \
from student s1, student smax, mark m1, mark mmax \
where s1.snum = m1.snum and \
      smax.snum = mmax.snum and \
      m1.deptcode = 'CS' and \
      m1.cnum = 240 and \
      mmax.deptcode = 'CS' and \ 
      mmax.cnum = 240 and \
      not exists (select * from mark where mark.deptcode = 'CS' and mark.cnum = 240 and mmax.grade < mark.grade) and \
      m1.grade >= mmax.grade - 3
      


*/


/*
4. The number and name of students who have completed two years, who
have a final grade of at least 85 in every computer science course that
they have taken, and who have always been taught by a professor in the
computer science (CS) department.
*/
-- there does not exist a mark in a CS course where he got less than 85.
-- there does not exist a prof who taught them that wasnt in the CS dept
-- the above 2 statements can be replace by a (= all) or (<> all) operator

select distinct s.snum as StudentNumber, s.sname as StudentName 
from student s
where s.year > 2 and -- Could be s.year = 3 
      not exists (select * 
	          from mark 
		  where mark.snum = s.snum and		
		  mark.deptcode = 'CS' and
		  mark.grade < 85) and
     not exists (select * 
	         from professor, class, enrollment 
		 where professor.deptcode <> 'CS' and
		 class.pnum = professor.pnum and 
		 enrollment.snum = s.snum and
		 enrollment.deptcode = class.deptcode and
		 enrollment.cnum  = class.cnum and
		 enrollment.term = class.term and
		 enrollment.section = class.section);
		
      
/*

select distinct s.snum as StudentNumber, s.sname as StudentName \
from student s \
where s.year > 2 and \ 
      not exists (select * \
	          from mark \
		  where mark.snum = s.snum and \	
		  mark.deptcode = 'CS' and \
		  mark.grade < 85) and \
     not exists (select * \
	         from professor, class, enrollment \
		 where professor.deptcode <> 'CS' and \
		 class.pnum = professor.pnum and \
		 enrollment.snum = s.snum and \
		 enrollment.deptcode = class.deptcode and \
		 enrollment.cnum  = class.cnum and \
		 enrollment.term = class.term and \
		 enrollment.section = class.section)




*/






/*
5. A sorted list of all departments who do not have a professor currently (SORTED LIST HMM.)
teaching a course offered by a different department.
*/

-- there does not exist a prof in a department that is teaching a different department code CURRENTLY
-- since CURRENTLY is involved, we need to use the mark table.

-- ALSO PLEASE SORT BY DEPTCODE
-- FIGURE OUT HOW TO SORT.

select department.deptcode as DepartmentCode, department.deptname as DepartmentName 
from department, professor
where professor.deptcode = department.deptcode and
	not exists (select * from class 
		    where (not exists (select * from mark 		
				       where class.deptcode = mark.deptcode and 
					     class.cnum = mark.cnum and 
					     class.term = mark.term and
					     class.section = mark.section) and
			   class.pnum = professor.pnum and 
			   class.deptcode <> professor.deptcode))
order by department.deptcode;
		  
		  

/*


select department.deptcode as DepartmentCode, department.deptname as DepartmentName \
from department, professor \
where professor.deptcode = department.deptcode and \
	not exists (select * from class  \
		    where (not exists (select * from mark \		
				       where class.deptcode = mark.deptcode and \ 
					     class.cnum = mark.cnum and \
					     class.term = mark.term and \
					     class.section = mark.section) and \
			   class.pnum = professor.pnum and \
			   class.deptcode <> professor.deptcode)) \
order by department.deptcode



*/		  
		  

/*
6. 

For each pair of classes for the same course that were taught in the same
term, and that were also taught by different professors: the minimum
grades obtained and the maximum grades obtained. In addition to these
four values, each result should include the number and name of each professor,
as well as the identifying attributes for each class.

*/

select distinct
	     class1.deptcode as classDepartmentCode,
	     class1.cnum as classCourseNumber,  
	     class1.term as classTerm, 
	     class1.section as class1Section,
	     class2.section as class2Section,
	     prof1.pnum as Professor1Number,
	     prof1.pname as Professor1Name ,
	     prof2.pnum as Professor2Number,
	     prof2.pname as Professor2Name,
	     markMin1.grade as  gradeMin1, 
	     markMax1.grade as gradeMax1, 
	     markMin2.grade as gradeMin2, 
	     markMax2.grade as gradeMax2
from 	   class class1, 
	   class class2, 
	   mark markMin1,
	   mark markMax1, 
	   mark markMin2,
	   mark markMax2, 
	   professor prof1, 
	   professor prof2
where	      class1.deptcode = class2.deptcode and 
	      class1.cnum = class2.cnum and 
	      class1.term = class2.term and 
	      class1.pnum <> class2.pnum and
	      prof1.pnum = class1.pnum and
	      prof2.pnum = class2.pnum and
	      
	      markMin1.deptcode = class1.deptcode and
	      markMin1.cnum = class1.cnum and 
	      markMin1.term = class1.term and
	      markMin1.section = class1.section and
	      
	      markMax1.deptcode = class1.deptcode and
	      markMax1.cnum = class1.cnum and 
	      markMax1.term = class1.term and
	      markMax1.section = class1.section and
	      
	      markMin2.deptcode = class2.deptcode and
	      markMin2.cnum = class2.cnum and 
	      markMin2.term = class2.term and
	      markMin2.section = class2.section and
	      
	      markMax2.deptcode = class2.deptcode and
	      markMax2.cnum = class2.cnum and 
	      markMax2.term = class2.term and
	      markMax2.section = class2.section and
	 
	      not exists (select * from mark markClass1 
				    where markClass1.deptcode = class1.deptcode and
					  markClass1.cnum = class1.cnum and
					  markClass1.term = class1.term and
					  markClass1.section = class1.section and
					  ((markClass1.grade > markMax1.grade) or (markClass1.grade < markMin1.grade)) ) and
						
	      not exists (select * from mark markClass2
				    where markClass2.deptcode = class2.deptcode and
					  markClass2.cnum = class2.cnum and
				          markClass2.term = class2.term and
					  markClass2.section = class2.section and
					  ((markClass2.grade > markMax2.grade) or (markClass2.grade < markMin2.grade)) ) ;
		
/*

select distinct \
	     class1.deptcode as classDepartmentCode, \
	     class1.cnum as classCourseNumber,  \
	     class1.term as classTerm, \
	     class1.section as class1Section, \
	     class2.section as class2Section, \
	     prof1.pnum as Professor1Number, \
	     prof1.pname as Professor1Name , \
	     prof2.pnum as Professor2Number, \
	     prof2.pname as Professor2Name, \
	     markMin1.grade as  gradeMin1, \
	     markMax1.grade as gradeMax1, \
	     markMin2.grade as gradeMin2, \
	     markMax2.grade as gradeMax2 \
from 	   class class1, \
	   class class2, \
	   mark markMin1, \
	   mark markMax1, \
	   mark markMin2, \
	   mark markMax2, \
	   professor prof1, \ 
	   professor prof2 \
where	      class1.deptcode = class2.deptcode and \ 
	      class1.cnum = class2.cnum and \
	      class1.term = class2.term and \
	      class1.pnum <> class2.pnum and \
	      prof1.pnum = class1.pnum and \
	      prof2.pnum = class2.pnum and \
	      markMin1.deptcode = class1.deptcode and \
	      markMin1.cnum = class1.cnum and \
	      markMin1.term = class1.term and \
	      markMin1.section = class1.section and \
	      markMax1.deptcode = class1.deptcode and \
	      markMax1.cnum = class1.cnum and \
	      markMax1.term = class1.term and \
	      markMax1.section = class1.section and \
	      markMin2.deptcode = class2.deptcode and \
	      markMin2.cnum = class2.cnum and  \
	      markMin2.term = class2.term and \
	      markMin2.section = class2.section and \
	      markMax2.deptcode = class2.deptcode and \
	      markMax2.cnum = class2.cnum and \ 
	      markMax2.term = class2.term and \
	      markMax2.section = class2.section and \
	      not exists (select * from mark markClass1 \
				    where markClass1.deptcode = class1.deptcode and \
					  markClass1.cnum = class1.cnum and \
					  markClass1.term = class1.term and \
					  markClass1.section = class1.section and \
 					  ((markClass1.grade > markMax1.grade) or (markClass1.grade < markMin1.grade)) ) and \
	      not exists (select * from mark markClass2 \
				    where markClass2.deptcode = class2.deptcode and \
					  markClass2.cnum = class2.cnum and \
				          markClass2.term = class2.term and \
					  markClass2.section = class2.section and \
					  ((markClass2.grade > markMax2.grade) or (markClass2.grade < markMin2.grade)) ) 








*/



/*

7. Pairs of distinct professors such that whenever the first one teaches a class
in a particular term the second also teaches a class for the same course
in the same term. Report a professor number and name for both the
professors.  (its an if statement => not ) 

a => b === not a or b === not (a and not b)

*/


-- distinct profs





--final solution
select distinct prof1.pnum as Prof1Number, prof1.pname as Prof1Name, prof2.pnum as Prof2Number, prof2.pname as Prof2Name
from professor prof1, professor prof2, class class1, class class2
where prof1.pnum <> prof2.pnum and 
((not exists (select * from class 
		where prof1.pnum = class.pnum and 
		      class2.pnum = prof2.pnum and 
		      class.term = class2.term)) or ( 
prof1.pnum = class1.pnum and 
prof2.pnum = class2.pnum and
class1.term = class2.term and
class1.deptcode = class2.deptcode and 
class1.cnum = class2.cnum))







/*


select distinct prof1.pnum as Prof1Number, prof1.pname as Prof1Name, prof2.pnum as Prof2Number, prof2.pname as Prof2Name \
from professor prof1, professor prof2, class class1, class class2 \
where prof1.pnum <> prof2.pnum and  \
((not exists (select * from class  \
		where prof1.pnum = class.pnum and \
		      class2.pnum = prof2.pnum and \
		      class.term = class2.term)) or ( \ 
prof1.pnum = class1.pnum and \
prof2.pnum = class2.pnum and \
class1.term = class2.term and \
class1.deptcode = class2.deptcode and \ 
class1.cnum = class2.cnum))



*/



/*
Queries that may use aggregation in SQL

The course number and total enrollment count for all of its classes of each
course. Also, include only those course numbers for courses with a total
enrollment count among the three lowest such counts. (Note that one
possible result could 

be:
{[CS, 348, 120], [CS, 446, 120], [CS, 341, 105], [CS, 245, 120], [CS, 234, 121]}.

Also note that all classes, past and ongoing, need to be considered.)

*/


select course.deptcode, course.cnum, COUNT(*) as cnt 
from class, course
where class.deptcode = course.deptcode 
      and class.cnum = course.cnum
group by course.deptcode, course.cnum
having COUNT(*) <=  (
			select COUNT(*) 
			from class, course
			where class.deptcode = course.deptcode 
			and class.cnum = course.cnum
			group by course.deptcode, course.cnum
			order by COUNT(*) asc
			limit 1 offset 2		
			) ;






/*


select course.deptcode, course.cnum, COUNT(*) as cnt \
from class, course \
where class.deptcode = course.deptcode \ 
      and class.cnum = course.cnum \
group by course.deptcode, course.cnum \
having COUNT(*) <=  ( \
			select COUNT(*) \ 
			from class, course \
			where class.deptcode = course.deptcode \ 
			and class.cnum = course.cnum \
			group by course.deptcode, course.cnum \
			order by COUNT(*) asc \
			limit 1 offset ) 




*/


/*
9. The percentage of professors in pure math who have always taught no
more than a single course in any given term. (Note that a percentage
should be a number between 0 and 100.)
*/
/*
select  (profsInPureMathSingleCoursePerTerm.cnt/ profsInPureMath.cnt) * 100 
from    (select class.term, prof.pnum, COUNT(prof.pnum) as cnt
	 from class, professor prof
	 where class.pnum = prof.pnum and prof.deptcode='PM'
	 group by class.term, prof.pnum
	 having COUNT(prof.pnum) <= 1  
	) as  profsInPureMathSingleCoursePerTerm,
	(
	 select COUNT(*) as cnt
	 from professor prof
	 where prof.deptcode = 'PM'
	) as profsInPureMath;
*/

select  (profsInPureMathSingleCoursePerTerm.cnt/ profsInPureMath.cnt) * 100 
from    (select class.term, prof.pnum, COUNT(prof.pnum) as cnt
	 from class, professor prof
	 where class.pnum = prof.pnum and prof.deptcode='PM'
	 group by class.term, prof.pnum
	 having COUNT(prof.pnum) <= 1  
	) as  profsInPureMathSingleCoursePerTerm,
	(
	 select COUNT(*) as cnt
	 from professor prof
	 where prof.deptcode = 'PM'
	) as profsInPureMath;

	
	
/*

select (profsInPureMathSingleCoursePerTerm.cnt/ profsInPureMath.cnt) * 100 
from    (select class.term, prof.pnum, COUNT(prof.pnum) as cnt \
	from class, professor prof \
	where class.pnum = prof.pnum and prof.deptcode='PM' \
	group by class.term, prof.pnum \
	having COUNT(prof.pnum) <= 1 \
	) as  profsInPureMathSingleCoursePerTerm, \
	( \
	select COUNT(*) as cnt \
	from professor prof \
	where prof.deptcode = 'PM' \
	) as profsInPureMath 
	




*/

	





/*
10. The number of different third or fourth year students in each section of
each course taught by a pure math professor in past terms. 

The result should include the professor number, professor name, course number and
section, and should also be sorted first by the name of the professor,
then by the professor number, third by the course number, and finally by
section. 

(Note that a section is identified by a term and a section number.

Also assume that sorting by section means sorting by term and then by
section number. The result will therefore have a total of six columns.)
*/

select thirdYearProfile.pnum as pnum, 
       thirdYearProfile.pname as pname,
       thirdYearProfile.deptcode as deptcode,  
       thirdYearProfile.cnum as cnum, 
       thirdYearProfile.section as section, 
       thirdYearProfile.cnt as thirdYearStudents, 
       fourthYearProfile.cnt as fourthYearStudents
from (select prof.pnum as pnum, 
             prof.pname as pname, 
	     class.deptcode as deptcode,  
	     class.cnum as cnum, 
	     class.section as section,
	     count(*) as cnt 
      from  professor prof, class, enrollment enroll, student
      where prof.deptcode = "PM" and
	    class.pnum = prof.pnum and
            class.section = enroll.section and
	    class.deptcode = enroll.deptcode and
	    class.cnum  = enroll.cnum and 
            class.term  = enroll.term and
            student.year = 3
      group by prof.pnum, prof.pname, class.cnum, class.deptcode, class.section) as thirdYearProfile,
      (select prof.pnum as pnum, 
              prof.pname as pname,
	      class.deptcode as deptcode, 
	      class.cnum as cnum, 
	      class.section as section, 
	      count(*) as cnt 
      from professor prof, class, enrollment enroll, student
      where prof.deptcode = "PM" and
            class.pnum = prof.pnum and
            class.section = enroll.section and
            class.deptcode = enroll.deptcode and
            class.cnum  = enroll.cnum and 
            class.term  = enroll.term and
            student.year = 4
      group by prof.pnum, prof.pname, class.cnum,  class.deptcode, class.section) as fourthYearProfile
where thirdYearProfile.pnum = fourthYearProfile.pnum and
      thirdYearProfile.pname = fourthYearProfile.pname and
      thirdYearProfile.deptcode = fourthYearProfile.deptcode and
      thirdYearProfile.cnum = fourthYearProfile.cnum and
      thirdYearProfile.section = fourthYearProfile.section 
order by pname, pnum, cnum, section;


/*


select thirdYearProfile.pnum as pnum, 
       thirdYearProfile.pname as pname, \
       thirdYearProfile.deptcode as deptcode, \  
       thirdYearProfile.cnum as cnum, \
       thirdYearProfile.section as section, \
       thirdYearProfile.cnt as thirdYearStudents, \
       fourthYearProfile.cnt as fourthYearStudents \
from (select prof.pnum as pnum, \
             prof.pname as pname, \ 
	     class.deptcode as deptcode, \  
	     class.cnum as cnum, \
	     class.section as section, \
	     count(*) as cnt \
      from  professor prof, class, enrollment enroll, student \
      where prof.deptcode = 'PM' and \
	    class.pnum = prof.pnum and \
            class.section = enroll.section and \
	    class.deptcode = enroll.deptcode and \
	    class.cnum  = enroll.cnum and \
            class.term  = enroll.term and \
            student.year = 3 \
      group by prof.pnum, prof.pname, class.cnum, class.deptcode, class.section) as thirdYearProfile, \
      (select prof.pnum as pnum, \
              prof.pname as pname, \
	      class.deptcode as deptcode, \ 
	      class.cnum as cnum, \
	      class.section as section, \ 
	      count(*) as cnt \
      from professor prof, class, enrollment enroll, student \
      where prof.deptcode = 'PM' and \
            class.pnum = prof.pnum and \
            class.section = enroll.section and \
            class.deptcode = enroll.deptcode and \
            class.cnum  = enroll.cnum and \
            class.term  = enroll.term and \
            student.year = 4 \
      group by prof.pnum, prof.pname, class.cnum,  class.deptcode, class.section) as fourthYearProfile \
where thirdYearProfile.pnum = fourthYearProfile.pnum and \
      thirdYearProfile.pname = fourthYearProfile.pname and \
      thirdYearProfile.deptcode = fourthYearProfile.deptcode and \
      thirdYearProfile.cnum = fourthYearProfile.cnum and \
      thirdYearProfile.section = fourthYearProfile.section \ 
order by pname, pnum, cnum, section

*/


/*
11. The ratio of professors in pure math (PM) to professors in applied math
(AM) who have taught a class in which the average grade obtained in the
class was greater than 77.
*/

select (PMProf.cnt/AMProf.cnt) * 100 as RatioOfPureMathProfsToAppliedMathProfs 
from (select COUNT(*) as cnt
      from professor prof
      where prof.deptcode = "PM" and 
	exists( select class.deptcode, class.cnum, class.term, class.section, AVG(mark.grade) as averageGradeInClass
		from mark, class
		where   class.pnum = prof.pnum and 
			class.deptcode = mark.deptcode and
			class.cnum = mark.cnum and 
			class.term = mark.term and
			class.section = mark.section
	group by class.deptcode, class.cnum, class.term, class.section
	having averageGradeInClass > 77)) PMProf,
      (select COUNT(*) as cnt
       from professor prof
       where prof.deptcode = "AM" and 
       exists( select class.deptcode, class.cnum, class.term, class.section, AVG(mark.grade) as averageGradeInClass
		from mark, class
		where   class.pnum = prof.pnum and 
			class.deptcode = mark.deptcode and
			class.cnum = mark.cnum and 
			class.term = mark.term and
			class.section = mark.section
		group by class.deptcode, class.cnum, class.term, class.section
		having averageGradeInClass > 77)) AMProf;

/*

select (PMProf.cnt/AMProf.cnt) * 100 as RatioOfPureMathProfsToAppliedMathProfs \
from (select COUNT(*) as cnt \
      from professor prof \
      where prof.deptcode = 'PM' and \ 
	exists( \
		select class.deptcode, class.cnum, class.term, class.section, AVG(mark.grade) as averageGradeInClass \
		from mark, class \
		where   class.pnum = prof.pnum and \ 
			class.deptcode = mark.deptcode and \
			class.cnum = mark.cnum and \
			class.term = mark.term and \
			class.section = mark.section \
	group by class.deptcode, class.cnum, class.term, class.section \
	having averageGradeInClass > 77)) PMProf, \
      (select COUNT(*) as cnt \
       from professor prof \
       where prof.deptcode = 'AM' and \ 
       exists( \
		select class.deptcode, class.cnum, class.term, class.section, AVG(mark.grade) as averageGradeInClass \
		from mark, class \
		where   class.pnum = prof.pnum and \
			class.deptcode = mark.deptcode and \
			class.cnum = mark.cnum and \
			class.term = mark.term and \
			class.section = mark.section \
		group by class.deptcode, class.cnum, class.term, class.section \
 		having averageGradeInClass > 77)) AMProf


*/


/*
12. For the current term (THE CURRNET TERM), report how many courses there are in the schedule
with a particular number of classes. For example an output
{[5, 1], [4, 2], [1, 5]}
indicates that there are 5 courses with a single class (section), 4 courses
with 2 classes, and 1 course with 5 classes scheduled in the curent term.

(NOTE: IT IS THE CURRENT TERM!!!)
*/
 
 

 
select COUNT(*) as NumberOfCourses, courseWithCount.classCount as NumberOfClassesForTheCourse
from (select course.deptcode, course.cnum, COUNT(*) as classCount
      from course, class, mark
      where not exists (select * 	--Check its a current class
                        from mark 
		        where mark.term = class.term and 
			      mark.section = class.section and 
			      mark.cnum = class.cnum and
			      mark.deptcode = class.deptcode
		       ) and 
      class.deptcode = course.deptcode and 
      class.cnum = course.cnum
      group by course.deptcode, course.cnum) courseWithCount
group by courseWithCount.classCount;


/*


select COUNT(*) as NumberOfCourses, courseWithCount.classCount as NumberOfClassesForTheCourse\
from (select course.deptcode, course.cnum, COUNT(*) as classCount \
      from course, class, mark \
      where not exists (select * \ 	
                        from mark \
		        where mark.term = class.term and \ 
			      mark.section = class.section and \ 
			      mark.cnum = class.cnum and \
			      mark.deptcode = class.deptcode \
		       ) and \ 
      class.deptcode = course.deptcode and \ 
      class.cnum = course.cnum \
      group by course.deptcode, course.cnum) courseWithCount \
group by courseWithCount.classCount




*/



















/*

I understand the point of GROUP BY x

But how does GROUP BY x, y work, and what does it mean?

Group By X means put all those with the same value for X in the one group.

Group By X, Y means put all those with the same values for 
both X and Y in the one group.

To illustrate using an example, let's say we have the 
following table, to do with who is attending what subject at a university:

Table: Subject_Selection

Subject   Semester   Attendee
---------------------------------
ITB001    1          John
ITB001    1          Bob
ITB001    1          Mickey
ITB001    2          Jenny
ITB001    2          James
MKB114    1          John
MKB114    1          Erica

When you use a group by on the subject column only; say:

select Subject, Count(*)
from Subject_Selection
group by Subject
You will get something like:

Subject    Count
------------------------------
ITB001     5
MKB114     2
...because there are 5 entries for ITB001, and 2 for MKB114

If we were to group by two columns:

select Subject, Semester, Count(*)
from Subject_Selection
group by Subject, Semester
we would get this:

Subject    Semester   Count
------------------------------
ITB001     1          3
ITB001     2          2
MKB114     1          2
This is because, when we group by two columns, it is saying "Group them so that all of 
those with the same Subject and Semester are in the same group, and then 
calculate all the aggregate functions (Count, Sum, Average, etc.) 
for each of those groups". In this example, this is demonstrated 
by the fact that, when we count them, there are three people 
doing ITB001 in semester 1, and two doing it in semester 2. Both of 
the people doing MKB114 are in semester 1, so there is no row 
for semester 2 (no data fits into the group "MKB114, Semester 2")

Hopefully that makes sense.



Example 1

Calculates the average advance and the sum of the sales for each type of book:

select type, avg (advance), sum (total_sales) 
from titles 
group by type
Example 2

Groups the results by type, then by pub_id within each type:

select type, pub_id, avg (advance), sum (total_sales) 
from titles 
group by type, pub_id
Example 3

Calculates results for all groups, but displays 
only groups whose type begins with “p”:

select type, avg (price) 
from titles 
group by type 
having type like 'p%'
Example 4

Calculates results for all groups, but displays 
results for groups matching the multiple conditions in the having clause:

select pub_id, sum (advance), avg (price) 
from titles 
group by pub_id 
having sum (advance) > $15000 
and avg (price) < $10 
and pub_id > "0700"
Example 5

Calculates the total sales for each group (publisher) 
after joining the titles and publishers tables:

select p.pub_id, sum (t.total_sales)
from publishers p, titles t
where p.pub_id = t.pub_id
group by p.pub_id
Example 6

Displays the titles that have an advance of more than $1000 
and a price that is more than the average price of all titles:

select title_id, advance, price
from titles
where advance > 1000
having price > avg (price)


SELECT sub.*
  FROM (
        SELECT *
          FROM tutorial.sf_crime_incidents_2014_01
         WHERE day_of_week = 'Friday'
       ) sub
 WHERE sub.resolution = 'NONE'
 

 
 SELECT LEFT(sub.date, 2) AS cleaned_month,
       sub.day_of_week,
       AVG(sub.incidents) AS average_incidents
  FROM (
        SELECT day_of_week,
               date,
               COUNT(incidnt_num) AS incidents
          FROM tutorial.sf_crime_incidents_2014_01
         GROUP BY 1,2
       ) sub
 GROUP BY 1,2
 ORDER BY 1,2
 

 SELECT e1.last_name, e1.first_name,
  (SELECT MAX(salary)
   FROM employees e2
   WHERE e1.employee_id = e2.employee_id) subquery2
FROM employees e1;

The trick to placing a subquery in the select clause is that 
the subquery must return a single value. This is why an 
aggregate function such as the SUM, COUNT, MIN, or MAX 
function is commonly used in the subquery.


SELECT   JobTitle,
         AVG(VacationHours) AS AverageVacationHours
FROM     HumanResources.Employee
GROUP BY JobTitle
HAVING   AVG(VacationHours) > (SELECT AVG(VacationHours)
                               FROM   HumanResources.Employee)
 

 SELECT   JobTitle,
         MaritalStatus,
         AVG(VacationHours)
FROM     HumanResources.Employee AS E
GROUP BY JobTitle, MaritalStatus
HAVING   AVG(VacationHours) > 
            (SELECT AVG(VacationHours)
             FROM   HumanResources.Employee
             WHERE  HumanResources.Employee. MaritalStatus =
                    E.MaritalStatus)
	
Also, with the correlated query, only fields used in the GROUP BY 
can be used in the inner query.  For instance, for kicks and 
grins, I tried replacing MaritalStatus with Gender and 
got an error.

SELECT   JobTitle,
         MaritalStatus,
         AVG(VacationHours)
FROM     HumanResources.Employee AS E
GROUP BY JobTitle, MaritalStatus
HAVING   AVG(VacationHours) > 
            (SELECT AVG(VacationHours)
             FROM   HumanResources.Employee
             WHERE  HumanResources.Employee. Gender = 
                    E. Gender)
Is a broken query.  If you try to run it you’ll get the following error:

Column ‘HumanResources.Employee.Gender’ is invalid in the HAVING clause 
because it is not contained in either an aggregate 
function or the GROUP BY clause.


 
AGGREGATION QUERY EXAMPLES:

Group By Clause:
The SQL GROUP BY Clause is used to output a row across 
specified column values.  It is typically used in conjunction 
with aggregate functions such as SUM or Count to summarize 
values.  In SQL groups are unique combinations of fields.  
Rather than returning every row in a table, when values 
are grouped, only the unique combinations are returned.


The GROUP BY Clause is added to the SQL Statement after the 
WHERE Clause.  Here is an example where we are listing OrderID, 
excluding quantities greater than 100.

SELECT OrderID
FROM OrderDetails
WHERE Quantity <= 100
GROUP BY OrderID;

There are a couple of things to note.  First, the columns 
we want to summarize are listed, separated by commas, in 
the GROUP BY clause.  Second, this same list of columns 
must be listed in the select statement; 
otherwise the statement fails.

When this statement is run, not every filtered row is returned.  
Only unique combinations of OrderID are included in 
the result.  This statement is very similar to

SELECT DISTINCT OrderID
FROM OrderDetails
WHERE Quantity <= 100;
But there is a key difference.  

The DISTINCT modifier stops at outputting unique combination 
of rows, whereas with the GROUP BY statement, 
we can calculate values based on the underlying 
filtered rows for each unique combination.

In other words, using our example, with the GROUP BY, 
we can calculate the number or OrderDetails per order as following:

SELECT OrderID, COUNT(OrderID) as NumOrderDetails
FROM OrderDetails
GROUP BY OrderID;
COUNT is an example of an aggregate function, these are what 
really give the GROUP BY statement its special value.



Some of the most common aggregate functions include:

AVG(expression)	Calculate the average of the expression.
COUNT(expression)	Count occurrences of non-null values returned by the expression.
COUNT(*)	Counts all rows in the specified table.
MIN(expression)	Finds the minimum expression value.
MAX(expression)	Finds the maximum expression value.
SUM(expression)	Calculate the sum of the expression.
 

These functions can be used on their own on in conjunction 
with the GROUP BY clause.  On their own, they operate across the 
entire table; however, when used with GROUP BY, their 
calculations are “reset” each time the grouping changes.  
In this manner they act as subtotals.


SELECT OrderID, 
       AVG(UnitPrice * Quantity) as AverageOrderAmount
FROM OrderDetails
GROUP BY OrderID;

For the curious, since an average is calculated as the sum 
of the sample divided by the sample count, then using 
AVG in the above statement is the same as:

SELECT OrderID, 
       SUM(UnitPrice * Quantity) / COUNT(OrderID)  as AverageOrderAmount
FROM OrderDetails
GROUP BY OrderID;


The COUNT function is used when you need to know how many 
records exist in a table or within a group.  COUNT(*) will 
count every record in the grouping; whereas COUNT(expression) 
counts every record where expression’s result isn’t null.  
You can also use Distinct with COUNT to find the number 
of unique values within a group.

To find the number of OrderDetail Lines per order

SELECT OrderID, COUNT(OrderDetailID)
FROM OrderDetails
GROUP BY OrderID;
To find the number of unique orders per product

SELECT ProductID, COUNT(DISTINCT OrderID)
FROM OrderDetails
GROUP BY ProductID;




If we wanted to find all orders greater than $1000 we would write

SELECT OrderID, 
       SUM(UnitPrice * Quantity) as TotalPrice
FROM OrderDetails
GROUP BY OrderID
HAVING TotalPrice > 1000
ORDER BY TotalPrice DESC;



To hammer home HAVING, I want to show one last example.  
Here you’ll see the HAVING statement includes an 
aggregate function that isn’t in the SELECT list.

SELECT OrderID, 
       SUM(UnitPrice * Quantity) as TotalPrice
FROM OrderDetails
GROUP BY OrderID
HAVING AVG(UnitPrice * Quantity) > 500
ORDER BY TotalPrice DESC;


Subqueries are not generally allowed in aggregate functions. 
Instead, move the aggregate inside the subquery. 
In this case, you'll need an extra level of 
subquery because of the top 5:

SELECT c.CategoryName,
  (select sum(val)
   from (SELECT TOP 5 od2.UnitPrice*od2.Quantity as val
         FROM [Order Details] od2, Products p2
         WHERE od2.ProductID = p2.ProductID
         AND c.CategoryID = p2.CategoryID
         ORDER BY 1 DESC
        ) t
  )
FROM [Order Details] od, Products p, Categories c, Orders o 
WHERE od.ProductID = p. ProductID
AND p.CategoryID = c.CategoryID
AND od.OrderID = o.OrderID
AND YEAR(o.OrderDate) = 1997
GROUP BY c.CategoryName, c.CategoryId


SELECT AVG(B.TotalBonus)
FROM   (SELECT   TerritoryID,
                 SUM(Bonus) AS TotalBonus
        FROM     Sales.SalesPerson
        GROUP BY TerritoryID) AS B
	
	

SELECT Employees.LastName, COUNT(Orders.OrderID) AS NumberOfOrders
FROM Orders
INNER JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID
WHERE LastName = 'Davolio' OR LastName = 'Fuller'
GROUP BY LastName
HAVING COUNT(Orders.OrderID) > 25;


"An aggregate may not appear in the WHERE clause unless it 
is in a subquery contained in a HAVING clause or a select list, 
and the column being aggregated is an outer reference"


Show all customer ids and number of orders for those who 
have 5 or more orders (and NULL for others):

SELECT o.customerid
     , ( SELECT COUNT( o.customerid )
         FROM account a
         WHERE a.customerid = o.customerid
           AND COUNT( o.customerid ) >= 5
        )
        AS cnt
FROM orders o
GROUP BY o.customerid ;




down vote
accepted
All columns in the SELECT clause that do not have an aggregate need to be in the GROUP BY

Good:

SELECT col1, col2, col3, MAX(col4)
...
GROUP BY col1, col2, col3
Also good:

SELECT col1, col2, col3, MAX(col4)
...
GROUP BY col1, col2, col3, col5, col6
No other columns = no GROUP BY needed

SELECT MAX(col4)
...
Won't work:

SELECT col1, col2, col3, MAX(col4)
...
GROUP BY col1, col2
Pointless:

SELECT col1, col2, col3, MAX(col4)
...
GROUP BY col1, col2, col3, MAX(col4)
Having an aggregate (MAX etc) with other columns without a GROUP BY makes no sense because the query becomes ambiguous.

*/


