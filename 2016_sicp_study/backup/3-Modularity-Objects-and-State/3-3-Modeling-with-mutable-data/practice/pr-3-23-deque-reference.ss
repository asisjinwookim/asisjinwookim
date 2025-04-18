;; referenced by : https://wqzhang.wordpress.com/2009/07/19/sicp-exercise-3-23/
;; internal procedures
(define (front-ptr deque) (car deque))
(define (rear-ptr deque) (cdr deque))
(define (set-front-ptr! deque item) (set-car! deque item))
(define (set-rear-ptr! deque item) (set-cdr! deque item))

(define (make-deque) (cons '() '()))
(define (empty-deque? deque)
  (and (null? (front-ptr deque))
	   (null? (rear-ptr deque))))
(define (print-deque deque)
  (define (make-printable-list q)
	(if (null? q)
		'()
		(cons (car q)
			  (make-printable-list (cddr q)))))
  (newline)
  (display (make-printable-list (front-ptr deque))))
(define (rear-insert-deque! deque item)
  (let ((new-pair (cons item (cons '() '()))))
	(cond ((empty-deque? deque)
		   (set-front-ptr! deque new-pair)
		   (set-rear-ptr! deque new-pair))
		  (else
		   (set-car! (cdr new-pair) (rear-ptr deque))
		   (set-cdr! (cdr (rear-ptr deque)) new-pair)
		   (set-rear-ptr! deque new-pair)))))
(define (front-insert-deque! deque item)
  (let ((new-pair (cons item (cons '() '()))))
	(cond ((empty-deque? deque)
		   (set-front-ptr! deque new-pair)
		   (set-rear-ptr! deque new-pair))
		  (else
		   (set-cdr! (cdr new-pair) (front-ptr deque))
		   (set-car! (cdr (front-ptr deque)) new-pair)
		   (set-front-ptr! deque new-pair)))))
(define (front-deque deque)
  (if (empty-deque? deque)
	  (error "FRONT called with an empty deque" deque)
	  (car (front-ptr deque))))
(define (rear-deque deque)
  (if (empty-deque? deque)
	  (error "REAR called with an empty deque" deque)
	  (car (rear-ptr deque))))
(define (front-delete-deque! deque)
  (cond ((empty-deque? deque)
		 (error "FRONT-DELETE! called with an empty deque" deque))
		((eq? (front-ptr deque) (rear-ptr deque))
		 (set-front-ptr! deque '())
		 (set-rear-ptr! deque '()))
		(else
		 (set-front-ptr! deque (cddr (front-ptr deque)))
		 (set-car! (cdr (front-ptr deque)) '()))))
(define (rear-delete-deque! deque)
  (cond ((empty-deque? deque)
		 (error "REAR-DELETE! called with an empty deque" deque))
		((eq? (front-ptr deque) (rear-ptr deque))
		 (set-front-ptr! deque '())
		 (set-rear-ptr! deque '()))
		(else
		 (set-rear-ptr! deque (cadr (rear-ptr deque)))
		 (set-cdr! (cdr (rear-ptr deque)) '()))))

										; test
(define q1 (make-deque))
(front-insert-deque! q1 'a)
(print-deque q1)
										; (a)
(front-insert-deque! q1 'b)
(print-deque q1)
										; (b a)
(rear-insert-deque! q1 'x)
(print-deque q1)
										; (b a x)
(rear-insert-deque! q1 'y)
(print-deque q1)
										; (b a x y)
(rear-delete-deque! q1)
(print-deque q1)
										; (b a x)
(front-delete-deque! q1)
(print-deque q1)
										; (a x)
(front-delete-deque! q1)
(print-deque q1)
										; (x)
(front-delete-deque! q1)
(print-deque q1)
										; ()
(empty-deque? q1)
										;Value: #t

(define q2 (make-deque))
(rear-insert-deque! q2 1)
(print-deque q2)
										; (1)
(front-insert-deque! q2 3)
(print-deque q2)
										; (3 1)
(front-insert-deque! q2 5)
(print-deque q2)
										; (5 3 1)
(front-deque q2)
										;Value: 5
(rear-deque q2)
										;Value: 1
(front-delete-deque! q2)
(print-deque q2)
										; (3 1)
(front-deque q2)
										;Value: 3
(rear-deque q2)
										;Value: 1
(rear-delete-deque! q2)
(print-deque q2)
										; (3)
(front-deque q2)
										;Value: 3
(rear-deque q2)
										;Value: 3
(empty-deque? q2)
										;Value: #f
(rear-delete-deque! q2)
(print-deque q2)
										; ()
(empty-deque? q2)
;Value: #t
