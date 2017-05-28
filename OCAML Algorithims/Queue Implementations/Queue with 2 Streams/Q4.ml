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
    
    let rec rev (stream : 'a stream) : 'a stream =
        lazy(
            match Lazy.force stream with
            | Nil -> acc
            | Cons(hd, tl) -> aux(tl)(lazy(Cons(hd, acc)))
            ) 
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

end