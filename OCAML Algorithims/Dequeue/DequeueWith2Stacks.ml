(*
the simplest solution would be to use one stack as the head of the dequeue, and one as the tail. 
The endequeue operations would just be a push to the respective stack, and the dedequeue 
operations would just be a pop on the respective stack.

However, if the stack you want to dedequeue from is empty, you'd have to pop each element from the 
other stack and push it back to the stack you want to dedequeue from, and then dedequeue the last one. 
That's not really good performance, so the overall performance of this implementation strongly
 depends on the workload. If your workload is a balanced number of front/back endequeue and dedequeue operations, 
 then this will be really really fast. But if your workload consists of a lot of alternating head-dedequeues
  and tail-dedequeues while the dequeue is large, then this will obviously be a bad approach.



*)

module type Stack = sig
 type 'a stack =
    Empty
  | Cons of 'a * 'a stack ;;

  val empty : 'a stack
  val is_empty : 'a stack -> bool

  val push : 'a -> 'a stack -> 'a stack
  val front : 'a stack -> 'a option
  val pop : 'a stack -> 'a stack option
end

module MyStack : Stack = struct

   type 'a stack =
    Empty
  | Cons of 'a * 'a stack ;;

let empty = Empty;;
let is_empty stack =  
  match stack with 
  | Empty -> true
  | _ -> false ;;

let push element stack = 
  match stack with 
  | Empty -> Cons(element, Empty)
  | _ -> Cons(element, stack);;
  
let pop stack = 
  match stack with
  | Empty -> None
  | Cons(a, restOfStack) -> Some restOfStack ;;

let front stack =
  match stack with 
  | Empty -> None
  | Cons(front, restOfStack) -> Some front ;;
end


module type Dequeue = sig
  type 'a stack =
    Empty
  | Cons of 'a * 'a stack ;;

 type 'a dequeue = int * 'a MyStack.stack * int * 'a MyStack.stack

  val empty : 'a dequeue
  val is_empty : 'a dequeue -> bool
 
  val snoc : 'a -> 'a dequeue -> 'a dequeue
  val cons : 'a -> 'a dequeue -> 'a dequeue
  val first : 'a dequeue -> 'a option
  val last: 'a dequeue -> 'a option
  val pop : 'a dequeue -> 'a dequeue option
  val eject: 'a dequeue -> 'a dequeue option
end

module MyDequeue : Dequeue = struct

  type 'a stack = 'a MyStack.stack;;
  
  type 'a dequeue = 'a MyStack.stack * 'a MyStack.stack
  (*Left queue is inbox, other queue is outbox*)
let empty = (0, MyStack.Empty, 0, MyStack.Empty);;

let reverseHalf dequeue =
  let aux from putinto count =
    if(count = 0) then (from, putinto)
    else begin
    let front = MyStack.front(from) in
    let newFrom = MyStack.pop(from) in
    let newTo = MyStack.push(front)(putinto) in
    aux(newFrom)(newTo)(count-1) 
    end in
  match dequeue with 
  | (sizeI, inbox, sizeO, outbox) when sizeI = 0 -> 
      let newSize = sizeO/2 in 
      let (newInbox, newOutbox) = aux(outbox)(inbox)(newSize) in (newSize, newInbox, sizeO-newSize, newOutbox) 
  | (sizeI, inbox, sizeO, outbox) when sizeO = 0 ->
      let newSize = sizeI/2 in 
      let (newInbox, newOutbox) = aux(outbox)(inbox)(newSize) in (sizeI - newSize, newInbox, newSize, newOutbox) ;;
  

let is_empty dequeue =  
  match dequeue with 
  | (_, MyStack.Empty, _, MyStack.Empty) -> true
  | _ -> false ;;

let first dequeue = 
  match dequeue with 
  | (MyStack.Empty, MyStack.Empty) -> None
  | (a, b) -> MyStack.first(b)

let last dequeue = 
   match dequeue with 
  | (MyStack.Empty, MyStack.Empty) -> None
  | (a, b) -> MyStack.first(a)

let snoc element dequeue = 
  match dequeue with 
  | (MyStack.Empty, MyStack.Empty) -> (MyStack.Empty, MyStack.push(element)(MyStack.Empty))
  | (inbox, outbox) -> (MyStack.push(element)(MyStack.Empty), outbox);;

let cons element dequeue = 
  match dequeue with 
  |   | (MyStack.Empty, MyStack.Empty) -> (MyStack.Empty, MyStack.push(element)(MyStack.Empty))

let pop dequeue = 
  match dequeue with
  |  (MyStack.Empty, MyStack.Empty) -> None
  |  (inbox, outbox)-> Some restOfStack ;;

let eject queue

end