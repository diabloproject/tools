# Local connection protocol

## Header

Header starts with a magic number and a version number.
For now, bytes are: 0x12233445 for the magic number and 0x00000001 for the version number.
Then, goes the command id. For upload file it's 0x00000001.
Next, there are 4 bytes for field ID, then field value.
For example, to upload a file, you need 5 fields:
- dest_path_offset
- dest_path_length
- file_content_offset
- file_content_length

So, field IDs are:
- dest_path_offset => 0x00000001
- dest_path_length => 0x00000002
- file_content_offset => 0x00000003
- file_content_length => 0x00000004

Then, each field has a value, in the header we are interested in offsets (from 0 to int_max), which are starting addresses in the dynamic buffer, and lengths, which are the number of bytes that field will occupy in the buffer.

For example, if the file we want to upload contains
```
Hello, world!
```
and destination path is `/tmp/hello`, then buffer can look like this (not guaranteed to be exactly like this):
b"Hello, world!/tmp/hello"
For such buffer upload header is:

| field_id |field_value|
|----------|-----------|
|0x00000001|0x0000000d |
|0x00000002|0x0000000a |
|0x00000003|0x00000001 |
|0x00000004|0x00000000 |
|0x00000005|0x0000000d |

And it's binary representation is:
`0x12233445 0x00000001 0x00000001 0x0000000d 0x00000002 0x0000000a 0x00000003 0x00000001 0x00000004 0x00000000 0x00000005 0x0000000d`
