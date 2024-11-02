## C Coding Style Conventions

Here is a list with some of the code conventions used by raylib:

Code element | Convention | Example
--- | :---: | ---
Defines | ALL_CAPS | `#define PLATFORM_DESKTOP`
Macros | ALL_CAPS | `#define MIN(a,b) (((a)<(b))?(a):(b))`
Variables | lowerCase | `int screen_width = 0;`, `float target_frame_time = 0.016f;`
Local variables | lowerCase | `Vector2 player_position = { 0 };`
Global variables | lowerCase | `bool window_ready = false;`
Constants | lowerCase | `const int max_value = 8;`
Pointers | MyType *pointer | `Texture2D *array = NULL;`
float values | always x.xf | `float gravity = 10.0f`
Operators | value1*value2 | `int product = value*6;`
Operators | value1/value2 | `int division = value/4;`
Operators | value1 + value2 | `int sum = value + 10;`
Operators | value1 - value2 | `int res = value - 5;`
Enum | TitleCase | `enum TextureFormat`
Enum members | ALL_CAPS | `PIXELFORMAT_UNCOMPRESSED_R8G8B8`
Struct | TitleCase | `struct Texture2D`, `struct Material`
Struct members | lowerCase | `texture.width`, `color.r`
Functions | TitleCase | `InitWindow()`, `LoadImageFromMemory()`
Functions params | lowerCase | `width`, `height`
Ternary Operator | (condition)? result1 : result2 | `printf("Value is 0: %s", (value == 0)? "yes" : "no");`

## Files and Directories Naming Conventions

  - Directories will be named using `snake_case`: `resources/models`, `resources/fonts`

  - Files will be named using `snake_case`: `main_title.png`, `cubicmap.png`, `sound.wav`

_NOTE: Avoid any space or special character in the files/dir naming!_
