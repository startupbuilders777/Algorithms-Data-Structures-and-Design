(* Run-length encoding of a list. 
# encode ["a";"a";"a";"a";"b";"c";"c";"a";"a";"d";"e";"e";"e";"e"];;
- : (int * string) list =
[(4, "a"); (1, "b"); (2, "c"); (2, "a"); (1, "d"); (4, "e")]

Run-length encoding (RLE) is a very simple form of lossless data 
compression in which runs of data (that is, sequences in which the 
same data value occurs in many consecutive data elements) are stored
 as a single data value and count, rather than as the original run. 
 This is most useful on data that contains many such runs. Consider, 
 for example, simple graphic images such as icons, line drawings, and 
 animations. It is not useful with files that don't have many runs as 
 it could greatly increase the file size.
*)

let encode list =
    let rec aux count acc = function
      | [] -> [] (* Can only be reached if original list is empty *)
      | [x] -> (count+1, x) :: acc
      | a :: (b :: _ as t) -> if a = b then aux (count + 1) acc t
                              else aux 0 ((count+1,a) :: acc) t in
    List.rev (aux 0 [] list);;
