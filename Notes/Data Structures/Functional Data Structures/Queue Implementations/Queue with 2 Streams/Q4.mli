
module type StreamType = sig
  type 'a stream_cell = Nil | Cons of 'a * 'a stream
  and 'a stream = 'a stream_cell Lazy.t
 
  val (++) : 'a stream -> 'a stream -> 'a stream  (* stream append *)
  val rev : 'a stream -> 'a stream
end
 
module Stream : StreamType
 
module type Queue = sig
  type 'a queue = int * 'a Stream.stream * int * 'a Stream.stream
 
  val empty : 'a queue
  val is_empty : 'a queue -> bool
 
  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end
 
module TSQueue : Queue