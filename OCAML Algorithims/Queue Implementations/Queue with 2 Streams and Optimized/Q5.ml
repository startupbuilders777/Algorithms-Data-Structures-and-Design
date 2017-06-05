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
    
    let rot (front: 'a Stream.stream) (back: 'a list): 'a Stream.stream = 
        let rec aux (front: 'a Stream.stream) (back: 'a list) (acc : 'a Stream.stream) : 'a Stream.stream =
        match (Lazy.force front) with
        | Stream.Nil -> 
            begin 
            match back with
            | [] -> acc
            | rhd::rtl -> aux(lazy Nil)(rtl)(lazy(Stream.Cons(rhd, acc)))
            end
        | Stream.Cons(fhd, ftl) ->
            begin 
            match back with
            | [] -> lazy(Stream.Cons(fhd, aux(ftl)([])(acc)))
            | rhd::rtl -> lazy(Stream.Cons(fhd, aux(ftl)(rtl)(lazy(Stream.Cons(rhd, acc)))))
            end
        in aux(front)(back)(lazy(Nil)) ;;
(*
    let snoc (element: 'a) (queue: 'a queue) : 'a queue = 
        match queue with 
        | (lenF, streamF, lenR, streamR) when lenR = lenF  -> 
            (lenF + lenR + 1, Stream.(++)(streamF)(Stream.rev(lazy(Stream.Cons(element, streamR)))), 0, lazy(Stream.Nil))
        | (lenF, streamF, lenR, streamR) -> 
            (lenF, streamF, lenR+1, lazy(Stream.Cons(element, streamR)));;

    let first (queue: 'a queue) : 'a option = 
        let getHead stream = 
            match Lazy.force stream with
            | Stream.Cons(hd, tl) -> Some hd
            | Stream.Nil -> None in
        match queue with
        | (lenF, streamF, _, _) when lenF > 0 -> getHead streamF
        | _ -> None ;;
        
        
    let rest (queue: 'a queue) : 'a queue option =        
        match queue with 
        | (0, streamF, lenR, streamR) -> None
        | (lenF, streamF, lenR, streamR) when lenF = lenR  ->
            begin
            match Lazy.force streamF with
            | Cons(hd, tl) -> Some (lenF + lenR - 1, Stream.(++)(tl)(Stream.rev(streamR)), 0, lazy(Stream.Nil))
            end
        | (lenF, streamF, lenR, streamR) when lenF > 0 ->
            begin 
            match Lazy.force streamF with
            | Cons (hd, tl) ->  Some (lenF - 1, tl, lenR, streamR)
            end;;

    let streamA : char Stream.stream = lazy(Stream.Cons('a', (lazy(Stream.Cons('b', lazy(Stream.Nil))))));;
    let streamB : char Stream.stream = lazy(Stream.Cons('c', (lazy(Stream.Cons('d', lazy(Stream.Nil))))));;

    let queueAB : char queue = (2, streamA, 2, streamB) 

    
    (*Helper Testing Methods*)
    let rec printQueue (queue : 'a queue) : unit = 
        match queue with
        | (lenF, streamF, lenR, streamR) when lenF > 0 ->
            begin
            match Lazy.force streamF with
            | Stream.Cons(hd, tl) -> Printf.printf "On Front: %c\n" hd ; printQueue (lenF - 1, tl, lenR, streamR) 
            end
        | (lenF, streamF, lenR, streamR) when lenR > 0 -> 
            begin
            match Lazy.force streamR with
            | Stream.Cons(hd, tl) -> Printf.printf "On Back: %c\n" hd ; printQueue (lenF, streamF, lenR - 1, tl)
            end
        | _ -> Printf.printf "end" ;;
 
    let get value =
        match value with 
        | Some x -> x;;
*)
end