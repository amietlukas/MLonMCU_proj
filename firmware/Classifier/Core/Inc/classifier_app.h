#ifndef CLASSIFIER_APP_H
#define CLASSIFIER_APP_H

#ifdef __cplusplus
extern "C" {
#endif

void classifier_init(void);
void classifier_process(void);  /* never returns */

#ifdef __cplusplus
}
#endif

#endif
