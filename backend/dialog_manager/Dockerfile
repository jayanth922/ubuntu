FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY . .

EXPOSE 8000
CMD ["node", "server.js"]