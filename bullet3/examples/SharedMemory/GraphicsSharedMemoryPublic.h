#ifndef GRAPHICS_SHARED_MEMORY_PUBLIC_H
#define GRAPHICS_SHARED_MEMORY_PUBLIC_H

#define GRAPHICS_SHARED_MEMORY_KEY 11347
///increase the SHARED_MEMORY_MAGIC_NUMBER whenever incompatible changes are made in the structures
///my convention is year/month/day/rev
//Please don't replace an existing magic number:
//instead, only ADD a new one at the top, comment-out previous one

#define GRAPHICS_SHARED_MEMORY_MAGIC_NUMBER 201904030
enum EnumGraphicsSharedMemoryClientCommand
{
	GFX_CMD_INVALID = 0,
	GFX_CMD_0,
	GFX_CMD_SET_VISUALIZER_FLAG,
	GFX_CMD_UPLOAD_DATA,
	GFX_CMD_REGISTER_TEXTURE,
	GFX_CMD_REGISTER_GRAPHICS_SHAPE,
	GFX_CMD_REGISTER_GRAPHICS_INSTANCE,
	GFX_CMD_SYNCHRONIZE_TRANSFORMS,
	GFX_CMD_REMOVE_ALL_GRAPHICS_INSTANCES,
	GFX_CMD_REMOVE_SINGLE_GRAPHICS_INSTANCE,
	GFX_CMD_CHANGE_RGBA_COLOR,
	GFX_CMD_GET_CAMERA_INFO,
	GFX_CMD_CHANGE_SCALING,
	//don't go beyond this command!
	GFX_CMD_MAX_CLIENT_COMMANDS,
};

enum EnumGraphicsSharedMemoryServerStatus
{
	GFX_CMD_SHARED_MEMORY_NOT_INITIALIZED = 0,
	//GFX_CMD_CLIENT_COMMAND_COMPLETED is a generic 'completed' status that doesn't need special handling on the client
	GFX_CMD_CLIENT_COMMAND_COMPLETED,
	GFX_CMD_CLIENT_COMMAND_FAILED,
	GFX_CMD_REGISTER_TEXTURE_COMPLETED,
	GFX_CMD_REGISTER_TEXTURE_FAILED,
	GFX_CMD_REGISTER_GRAPHICS_SHAPE_COMPLETED,
	GFX_CMD_REGISTER_GRAPHICS_SHAPE_FAILED,
	GFX_CMD_REGISTER_GRAPHICS_INSTANCE_COMPLETED,
	GFX_CMD_REGISTER_GRAPHICS_INSTANCE_FAILED,
	GFX_CMD_GET_CAMERA_INFO_COMPLETED,
	GFX_CMD_GET_CAMERA_INFO_FAILED,
	//don't go beyond 'CMD_MAX_SERVER_COMMANDS!
	GFX_CMD_MAX_SERVER_COMMANDS
};





#endif  //GRAPHICS_SHARED_MEMORY_PUBLIC_H
