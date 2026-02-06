#ifndef ACHO_DEVICES_H
#define ACHO_DEVICES_H

typedef struct acho_device_info {
	char name[256];
	char description[512];
	int  is_audio;  /* 0 = video, 1 = audio */
} acho_device_info;

/*
 * List available capture devices.
 * Allocates array of acho_device_info, caller must free with acho_devices_free().
 * Returns number of devices found, or negative error code.
 */
int acho_devices_list(acho_device_info **out);

void acho_devices_free(acho_device_info *devices);

#endif
