# Types for the exception exports used from the untyped Instructor package.

class InstructorError(Exception): ...

InstructorRetryException: type[InstructorError]
