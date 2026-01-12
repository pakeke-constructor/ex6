# Rich Layout Sizing

## 3 Options
- `size=N` - fixed lines (current)
- `ratio=N` - proportional, resizes w/ terminal
- `minimum_size=N` - floor constraint, use w/ ratio

## Best for input box
```python
Layout(..., ratio=1, minimum_size=3)
```
Flexible but won't collapse below 3 lines.

## Get terminal height dynamically:
```python
import shutil
_, h = shutil.get_terminal_size()
Layout(..., size=max(3, h//10))
```
