# Coordinate System Guide

## Camera Perspective (First Person)

```
                    UP (Sky)
                      ↑
                      |
        LEFT          |          RIGHT
        -X ←----------+-------→ +X
                      |
                      ↓ (Forward)
                     +Z

        (Away from viewer is -Z)
```

## Top-Down View (What You Define)

```
              -Z (Back/Away)
                    ↑
                    |
    -X (Left) ←-----+-----→ +X (Right)
                    |
                    ↓
              +Z (Forward)
```

## Example Zone - Rectangle

```
TOP-DOWN VIEW:

        (-20,-30) -------- (20,-30)
           |                  |
           |                  |
           |   GROUND FLOOR   |
           |                  |
           |                  |
        (-20, 30) -------- (20, 30)
```

### Points Array:

```python
[
    [-20, -30],  # Top-left
    [ 20, -30],  # Top-right
    [ 20,  30],  # Bottom-right
    [-20,  30]   # Bottom-left
]
```

## Complex Shapes

### L-Shaped Parking Lot

```
        (-30, -20) --- (0, -20)
           |              |
           |            (20, -20)
           |              |
           |              |
        (-30, 20) ---    (20, 20)
           |              |
           |              |
        (-30, 40) --- (20, 40)
```

Points:

```python
[
    [-30, -20],
    [  0, -20],
    [ 20, -20],
    [ 20,  20],
    [ 20,  40],
    [-30,  40],
    [-30,  20]
]
```

### Hexagon

```
              (0, -30)
             /        \
        (-26, -15)  (26, -15)
            |        |
        (-26, 15)  (26, 15)
             \        /
              (0, 30)
```

Points:

```python
[
    [  0, -30],
    [ 26, -15],
    [ 26,  15],
    [  0,  30],
    [-26,  15],
    [-26, -15]
]
```

## Guidelines

✓ **Positive X** = Right side of your map  
✓ **Negative X** = Left side of your map  
✓ **Positive Z** = Toward the camera/bottom of map  
✓ **Negative Z** = Away from camera/top of map

✓ **Minimum 3 points** needed  
✓ **Draw counterclockwise** or clockwise (doesn't matter)  
✓ **Coordinates can be 0-1000** or -100 to +100 (your choice)  
✓ **Y axis is auto** - you walk at 1.6m height

## Size Reference

When you spawn in the 3D world, you're 1.6 meters tall.

| Distance  | In-World Feel                   |
| --------- | ------------------------------- |
| 5 units   | ~5 meters (small room)          |
| 20 units  | ~20 meters (car length ~5m)     |
| 50 units  | ~50 meters (half a parking lot) |
| 100 units | ~100 meters (full lot)          |

## Real-World Parking Lot

**Typical parking space**: ~2.5m × 5m  
**Typical parking row**: 2.5m wide × 30-40m long  
**Typical lot**: 50m × 100m

To model this in units:

```python
# Parking space (2.5 × 5)
[
    [0, 0],
    [2.5, 0],
    [2.5, 5],
    [0, 5]
]

# Row of 8 spaces (2.5 × 40)
[
    [0, 0],
    [2.5, 0],
    [2.5, 40],
    [0, 40]
]

# Full lot (50 × 100)
[
    [-25, -50],
    [25, -50],
    [25, 50],
    [-25, 50]
]
```

## Testing Your Shape

1. **Draw on paper first**
2. **Pick two opposite corners** as reference (-X,-Z) and (+X,+Z)
3. **List all corners** going around: `[x1, z1], [x2, z2], ...`
4. **Add to Streamlit** and see it render
5. **Walk through it** to verify shapes look right

## Common Mistakes

❌ **Overlapping zones** → Zones may look weird (allowed but confusing)  
❌ **Self-intersecting polygon** → Boundary walls look broken  
❌ **Only 2 points** → Creates a line, not valid  
❌ **Very large coordinates** → 1000+ units = huge spaces, hard to navigate  
❌ **Copying negative X as negative Z** → Wrong shape

## Pro Tips

💡 **Start small**: Create a 20×20 square first  
💡 **Use round numbers**: 10, 20, 30 easier than 13.5, 27.2  
💡 **Offset zones**: Don't overlap (unless intentional)  
💡 **Color-code**: Different colors help distinguish zones  
💡 **Test movement**: Walk along boundaries to verify  
💡 **Use grids**: Think in multiples of 5 or 10

---

**Example to copy-paste:**

```python
# Simple 2-zone parking lot
zones = [
    {
        'id': 1,
        'name': 'Ground Floor - North',
        'points': [[-30, -40], [30, -40], [30, 0], [-30, 0]],
        'occupancy': 12
    },
    {
        'id': 2,
        'name': 'Ground Floor - South',
        'points': [[-30, 5], [30, 5], [30, 45], [-30, 45]],
        'occupancy': 15
    }
]
```

Paste this in your browser console after loading the app:

```javascript
window.updateZones(zones);
```

Done! Both zones appear in 3D. 🎉

## Troubleshooting

### Issue: Black screen

- Check browser console for errors (F12)
- Verify zones have 3+ points
- Ensure viewer.html is in correct directory

### Issue: Slow performance

- Reduce number of zones
- Use wireframe mode (F) to see complexity
- Clear browser cache and reload

### Issue: Camera stuck

- Press R to reset camera
- Ensure no obstacles blocking movement

### Issue: Points not appearing in 3D

- Verify coordinates are within reasonable range (-100 to 100)
- Ensure at least 3 points to form a valid polygon
- Check zone occupancy is being updated
