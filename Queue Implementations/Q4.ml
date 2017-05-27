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
 
module Stream : StreamType = struct
    type 'a stream_cell = Nil | Cons of 'a * 'a stream
    and 'a stream = 'a stream_cell Lazy.t
    
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
 
module TSQueue : Queue = struct 
    type 'a queue = int * 'a Stream.stream * int * 'a Stream.stream
    
    let empty = (0, lazy(Stream.Nil), 0, lazy(Stream.Nil));;
    
    let is_empty (queue : 'a queue) : bool = 
        match queue with
        | (0,_,0,_) -> true
        | _ -> false;;
    
    let snoc (element: 'a) (queue: 'a queue) : 'a queue = empty;;
    let first (queue: 'a queue) : 'a option = None;;
    let rest (queue: 'a queue) : 'a queue option = None;;

    let streamA : char Stream.stream = lazy(Stream.Cons('a', (lazy(Stream.Cons('b', lazy(Stream.Nil))))));;
    let streamB : char Stream.stream = lazy(Stream.Cons('c', (lazy(Stream.Cons('d', lazy(Stream.Nil))))));;

    let queueAB : char queue = (2, streamA, 2, streamB) 
end