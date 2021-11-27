#pragma once

#include <memory>

#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/python_stub.h>

namespace c10 {

// NOTE [What is PythonDispatcher?]
// A PythonDispatcher represents the type of a Tensor subclass that has
// a __torch_dispatch__ classmethod. Concretely, it holds the class as a
// PyObject* and a PyInterpreter* that says which python interpreter the
// class came from.
//
// See NOTE [dispatch_fn's type argument] for more details
struct C10_API PythonDispatcher {
  explicit PythonDispatcher(PyObject* type,
                            c10::impl::PyInterpreter* interpreter) noexcept;

  PythonDispatcher(const PythonDispatcher&) = delete;

  PythonDispatcher& operator=(const PythonDispatcher&) = delete;

  PythonDispatcher(PythonDispatcher&&) = delete;

  PythonDispatcher& operator=(PythonDispatcher&&) = delete;

  ~PythonDispatcher();

  void dispatch(const OperatorHandle& op, torch::jit::Stack* s) const;

  intrusive_ptr<TensorImpl> detach(const c10::TensorImpl* self) const;

  PyObject* type() const noexcept;

  c10::impl::PyInterpreter* interpreter() const noexcept;

 private:
  PyObject* type_;
  c10::impl::PyInterpreter* interpreter_;
};

C10_API const std::shared_ptr<PythonDispatcher>& getPythonDispatcher() noexcept;

C10_API void setPythonDispatcher(const std::shared_ptr<PythonDispatcher>& dispatcher) noexcept;

C10_API void popPythonDispatcher() noexcept;

C10_API bool hasPythonDispatcher() noexcept;

// Temporarily disables the Python dispatcher. Note that this guard disables
// only the thread-local dispatcher, not the Python dispatch key; this means
// instances of Tensor subclasses with `__torch_dispatch__` will continue to
// work.
class C10_API DisablePythonDispatcherGuard {
 public:
  DisablePythonDispatcherGuard() noexcept;

  DisablePythonDispatcherGuard(const DisablePythonDispatcherGuard&) = delete;

  DisablePythonDispatcherGuard& operator=(const DisablePythonDispatcherGuard&) = delete;

  DisablePythonDispatcherGuard(DisablePythonDispatcherGuard&&) = delete;

  DisablePythonDispatcherGuard& operator=(DisablePythonDispatcherGuard&&) = delete;

  ~DisablePythonDispatcherGuard();
};

} // namespace c10