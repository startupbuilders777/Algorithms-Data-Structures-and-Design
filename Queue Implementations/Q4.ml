(*

Provide an OCaml implementation of the Queue ADT using two streams (and their sizes) 
achieving persistence with all operations taking O(1) amortized time

*)

(*
The four fields in a queue tuple represent the length of the front segment, 
the front segment, the length of the rear segment, and the rear segment, respectively.
*)

module type StreamType = sig
  type 'a stream_cell = Nil | Cons of 'a * 'a stream
  and 'a stream = 'a stream_cell Lazy.t
 
  val (++) : 'a stream -> 'a stream -> 'a stream  (* stream append *)
  val rev : 'a stream -> 'a stream
end
 
let stream : char stream = lazy(Cons('a', (lazy(Cons('b', lazy(Nil))))));;


module Stream : StreamType = struct
    let rec (++) (stream1 : 'a stream) (stream2: 'a stream) : 'a stream =
        lazy (
        match Lazy.force stream1 with
        | Nil -> Lazy.force stream2
        | Cons(hd, tl) -> Cons(hd, tl ++ stream2)        
        );;
    
    let rev (stream : 'a stream) : 'a stream =
        let rec aux (stream: 'a stream) (acc : 'a stream) : 'a stream =
            match Lazy.force stream with
            | Nil -> acc
            | Cons(hd, tl) -> aux(tl)(lazy(Cons(hd, acc))) in
         aux(stream)(lazy(Nil));;
end
 
module type Queue = sig
  type 'a queue = int * 'a Stream.stream * int * 'a Stream.stream
 
  val empty : 'a queue
  val is_empty : 'a queue -> bool
 
  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end
 
module TSQueue : Queue

(*= struct
    let (++) = 2;
    let rev stream =
        let aux stream acc =    
            match stream with
            | Nil ->  
            |
        in 
        aux stream [] 



end
 
module type Queue = sig
  type 'a queue = int * 'a Stream.stream * int * 'a Stream.stream
 
  val empty : 'a queue
  val is_empty : 'a queue -> bool
 
  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end
 
module TSQueue : Queue = struct 
    let empty = (0, Stream.stream [], 0, Stream.stream [])
    let is_empty queue = 
        match queue with
        | (0,_,0,_) -> true
        | _ -> false
    
    let snoc element queue

end

*)