# Stage 1: Build the React application
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the application and build
COPY . .
RUN npm run build

# Stage 2: Serve the application using Apache
FROM httpd:alpine

# Copy the build output to Apache's document root
COPY --from=build /app/dist /usr/local/apache2/htdocs/

# Copy the custom Apache configuration
COPY httpd.conf /usr/local/apache2/conf/httpd.conf

# Expose port 80
EXPOSE 80

# Apache starts automatically in the foreground in the httpd:alpine image
