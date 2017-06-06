(*
Provide an OCaml implementation of the Queue ADT achieving persistence 
with all operations taking O(1) worst-case time, as described in lecture. Here is the signature.
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
  type 'a queue = 'a Stream.stream * 'a list * 'a Stream.stream
 
  val empty : 'a queue
  val is_empty : 'a queue -> bool
 
  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end
 
module RTQueue : Queue = struct 
    type 'a queue = 'a Stream.stream * 'a list * 'a Stream.stream
    
    let empty = (lazy(Stream.Nil), [], lazy(Stream.Nil));;
    
    let is_empty (queue : 'a queue) : bool = 
        match queue with
        | (lazy(Stream.Nil), [], lazy(Stream.Nil)) -> true
        | _ -> false;;
    
    let rec rot (front: 'a Stream.stream) (back: 'a list) (acc : 'a Stream.stream) : 'a Stream.stream =
        match (Lazy.force front) with
        | Stream.Nil -> 
            begin 
            match back with
            | [] -> acc
            | rhd::rtl -> rot(lazy Nil)(rtl)(lazy(Stream.Cons(rhd, acc)))
            end
        | Stream.Cons(fhd, ftl) ->
            begin 
            match back with
            | [] -> lazy(Stream.Cons(fhd, rot(ftl)([])(acc)))
            | rhd::rtl -> lazy(Stream.Cons(fhd, rot(ftl)(rtl)(lazy(Stream.Cons(rhd, acc)))))
            end;;

    let snoc (element: 'a) (queue: 'a queue) : 'a queue = 
        match queue with 
        | (streamF, listR, streamR) ->
            match (Lazy.force streamR) with
            | Stream.Nil -> let rotation = rot(streamF)(element::listR)(lazy(Stream.Nil)) in (rotation, [], rotation)
            | Cons(scheduleHd, scheduleTl) -> (streamF, element::listR, scheduleTl) 

    let first (queue: 'a queue) : 'a option = 
        let getHead stream = 
            match Lazy.force stream with
            | Stream.Cons(hd, tl) -> Some hd
            | Stream.Nil -> None in
        match queue with
        | (streamF, _, _) -> getHead streamF
        | _ -> None ;;
        
    let rec rest (queue: 'a queue) : 'a queue option =        
        match queue with 
        | (streamF, listR, streamR) ->
            match (Lazy.force streamF, Lazy.force streamR) with
            | (Stream.Nil, Stream.Nil) when listR = [] -> None
            | (_, Stream.Nil) -> let rotation = rot(streamF)(listR)(lazy(Stream.Nil)) in rest (rotation, [], rotation)
            | (Cons(fhd, ftl), Cons(scheduleHd, scheduleTl)) -> Some (ftl, listR, scheduleTl) 

    let streamA : char Stream.stream = lazy(Stream.Cons('a', (lazy(Stream.Cons('b', lazy(Stream.Nil))))));;
    let streamB : char Stream.stream = lazy(Stream.Cons('c', (lazy(Stream.Cons('d', lazy(Stream.Nil))))));;
    let listBackA = ['e'; 'f'; 'g'];;
    let queueAB = (lazy(Stream.Nil), [], lazy(Stream.Nil));;
    
    let rec printStream (stream: char Stream.stream) = 
        match Lazy.force stream with
            | Stream.Cons(hd,tl) -> Printf.printf "On Front: %c\n" hd ; printStream tl
            | Stream.Nil -> Printf.printf "End" ;;

    let get value =
        match value with 
        | Some x -> x;;        
end