include .env
export

deploy:
	gcloud run deploy ${AGENT_SERVICE_NAME} \
	--source. \
	--region ${GOOGLE_CLOUD_LOCATION} \
	--project ${GOOOCLE_CLOUD_PROJECT} \
	--allow-unauthenticated \
	--port=8000 \ 
	--set-env-vars ="DOODLE_CLOUD_PROJECT= ${GOOOCLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION =  ${GOOGLE_CLOUD_LOCATION}, GOOGLE_GENAI_USE_VERTEXAI=  {GOOGLE_GENAI_USE_VERTEXAI}"
