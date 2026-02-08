/**
 * WhatsApp Web Bridge for Aria
 *
 * Provides HTTP and WebSocket APIs for the Python backend
 * to interact with WhatsApp Web.
 */

const { Client, LocalAuth, MessageMedia } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
const express = require("express");
const cors = require("cors");
const { WebSocketServer } = require("ws");
const http = require("http");
const path = require("path");

const PORT = process.env.PORT || 3001;
const SESSION_PATH = process.env.SESSION_PATH || "./session";

// Express app
const app = express();
app.use(cors());
app.use(express.json({ limit: "50mb" }));

// HTTP server for both Express and WebSocket
const server = http.createServer(app);

// WebSocket server
const wss = new WebSocketServer({ server, path: "/ws" });

// Store connected WebSocket clients
const wsClients = new Set();

// Store latest QR code
let latestQR = null;

// Broadcast to all WebSocket clients
function broadcast(data) {
  const message = JSON.stringify(data);
  wsClients.forEach((client) => {
    if (client.readyState === 1) {
      // WebSocket.OPEN
      client.send(message);
    }
  });
}

// Initialize WhatsApp client
const client = new Client({
  authStrategy: new LocalAuth({
    dataPath: SESSION_PATH,
  }),
  puppeteer: {
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-accelerated-2d-canvas",
      "--no-first-run",
      "--no-zygote",
      "--disable-gpu",
    ],
  },
});

// Client ready state
let isReady = false;

// Track message IDs sent by the bot (via /send endpoint) to avoid echo loops
const botSentMessageIds = new Set();
// Auto-cleanup old IDs every 5 minutes
setInterval(() => {
  botSentMessageIds.clear();
}, 5 * 60 * 1000);

// QR Code event
client.on("qr", (qr) => {
  console.log("QR Code received. Scan with WhatsApp:");
  qrcode.generate(qr, { small: true });
  latestQR = qr;
  broadcast({ type: "qr", qr });
});

// Ready event
client.on("ready", () => {
  console.log("WhatsApp client is ready!");
  isReady = true;
  latestQR = null;
  broadcast({ type: "ready" });
});

// Authenticated event
client.on("authenticated", () => {
  console.log("WhatsApp authenticated");
  broadcast({ type: "authenticated" });
});

// Auth failure event
client.on("auth_failure", (msg) => {
  console.error("Authentication failed:", msg);
  broadcast({ type: "auth_failure", message: msg });
});

// Disconnected event
client.on("disconnected", (reason) => {
  console.log("WhatsApp disconnected:", reason);
  isReady = false;
  broadcast({ type: "disconnected", reason });
});

// Message event — use message_create to capture ALL messages including own (fromMe)
// This is needed because the bridge runs as the user's WhatsApp session,
// so the user's own messages only fire on message_create, not on message.
client.on("message_create", async (msg) => {
  try {
    // Skip status broadcasts
    if (msg.isStatus) return;

    // Skip messages that WE sent via the /send endpoint (bot replies)
    const msgSerializedId = msg.id._serialized;
    if (botSentMessageIds.has(msgSerializedId)) {
      botSentMessageIds.delete(msgSerializedId);
      return;
    }

    // For fromMe messages that aren't in botSentMessageIds,
    // these are messages the user typed on their phone — process them.
    // For non-fromMe messages, these are incoming from others — always process.

    const chat = await msg.getChat();
    const contact = await msg.getContact();

    const messageData = {
      type: "message",
      message: {
        id: msg.id,
        body: msg.body,
        type: msg.type,
        timestamp: msg.timestamp,
        fromMe: msg.fromMe,
        hasMedia: msg.hasMedia,
        isForwarded: msg.isForwarded,
        isStatus: msg.isStatus,
        isVoice: msg.type === "ptt",
      },
      from: {
        id: contact.id._serialized,
        name: contact.name,
        pushname: contact.pushname,
        isMe: contact.isMe,
        isGroup: contact.isGroup,
      },
      chatId: chat.id._serialized,
      isGroup: chat.isGroup,
    };

    // Get media URL if available
    if (msg.hasMedia) {
      try {
        const media = await msg.downloadMedia();
        if (media) {
          messageData.message.mediaData = media.data;
          messageData.message.mimetype = media.mimetype;
          messageData.message.filename = media.filename;
        }
      } catch (err) {
        console.error("Error downloading media:", err);
      }
    }

    // Get quoted message if reply
    if (msg.hasQuotedMsg) {
      try {
        const quoted = await msg.getQuotedMessage();
        messageData.message.quotedMsgId = quoted.id._serialized;
      } catch (err) {
        console.error("Error getting quoted message:", err);
      }
    }

    console.log(`[MSG] from=${contact.id._serialized} fromMe=${msg.fromMe} chat=${chat.id._serialized} body=${(msg.body || "").substring(0, 50)}`);
    broadcast(messageData);
  } catch (err) {
    console.error("Error processing message:", err);
  }
});

// Reaction event
client.on("message_reaction", async (reaction) => {
  broadcast({
    type: "reaction",
    messageId: reaction.msgId._serialized,
    reaction: reaction.reaction,
    userId: reaction.senderId,
  });
});

// WebSocket connection handler
wss.on("connection", (ws) => {
  console.log("WebSocket client connected");
  wsClients.add(ws);

  // Send current status
  ws.send(
    JSON.stringify({
      type: "status",
      ready: isReady,
      hasQR: !!latestQR,
    })
  );

  ws.on("close", () => {
    console.log("WebSocket client disconnected");
    wsClients.delete(ws);
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err);
    wsClients.delete(ws);
  });
});

// REST API Routes

// Status endpoint
app.get("/status", (req, res) => {
  res.json({
    ready: isReady,
    hasQR: !!latestQR,
  });
});

// QR code endpoint
app.get("/qr", (req, res) => {
  if (latestQR) {
    res.json({ qr: latestQR });
  } else if (isReady) {
    res.json({ message: "Already authenticated" });
  } else {
    res.status(404).json({ error: "QR code not available" });
  }
});

// Send message endpoint
app.post("/send", async (req, res) => {
  if (!isReady) {
    return res.status(503).json({ error: "WhatsApp not ready" });
  }

  const { chatId, content, media, mediaType, filename, quotedMessageId, sendAsVoice } =
    req.body;

  try {
    let messageOptions = {};

    // Handle quoted message (reply)
    if (quotedMessageId) {
      messageOptions.quotedMessageId = quotedMessageId;
    }

    let result;

    const doSend = async (opts) => {
      if (media) {
        const mediaMsg = new MessageMedia(
          mediaType || "application/octet-stream",
          media,
          filename
        );
        if (sendAsVoice) opts.sendAudioAsVoice = true;
        return await client.sendMessage(chatId, mediaMsg, opts);
      } else {
        return await client.sendMessage(chatId, content, opts);
      }
    };

    try {
      result = await doSend(messageOptions);
    } catch (sendErr) {
      // If quoting failed, retry without the quote
      if (quotedMessageId) {
        console.warn("Send with quote failed, retrying without quote:", sendErr.message);
        delete messageOptions.quotedMessageId;
        result = await doSend(messageOptions);
      } else {
        throw sendErr;
      }
    }

    // Track this message ID so we don't process it as an incoming message
    const sentId = result.id._serialized;
    botSentMessageIds.add(sentId);

    res.json({
      success: true,
      messageId: sentId,
    });
  } catch (err) {
    console.error("Send error:", err);
    res.status(500).json({ error: err.message });
  }
});

// React to message endpoint
app.post("/react", async (req, res) => {
  if (!isReady) {
    return res.status(503).json({ error: "WhatsApp not ready" });
  }

  const { messageId, reaction } = req.body;

  try {
    // Note: Reaction API may require specific whatsapp-web.js version
    res.json({ success: true });
  } catch (err) {
    console.error("React error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Typing indicator endpoint
app.post("/typing", async (req, res) => {
  if (!isReady) {
    return res.status(503).json({ error: "WhatsApp not ready" });
  }

  const { chatId, typing } = req.body;

  try {
    const chat = await client.getChatById(chatId);
    if (typing) {
      await chat.sendStateTyping();
    } else {
      await chat.clearState();
    }
    res.json({ success: true });
  } catch (err) {
    console.error("Typing error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get contact info endpoint
app.get("/contact/:id", async (req, res) => {
  if (!isReady) {
    return res.status(503).json({ error: "WhatsApp not ready" });
  }

  try {
    const contact = await client.getContactById(req.params.id);
    res.json({
      id: contact.id._serialized,
      name: contact.name,
      pushname: contact.pushname,
      isMe: contact.isMe,
      isGroup: contact.isGroup,
      isBlocked: contact.isBlocked,
    });
  } catch (err) {
    console.error("Contact error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Download media endpoint
app.post("/download", async (req, res) => {
  if (!isReady) {
    return res.status(503).json({ error: "WhatsApp not ready" });
  }

  const { mediaKey } = req.body;

  try {
    // Media download would need message reference
    // This is a simplified endpoint
    res.status(501).json({ error: "Not implemented" });
  } catch (err) {
    console.error("Download error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get chats endpoint
app.get("/chats", async (req, res) => {
  if (!isReady) {
    return res.status(503).json({ error: "WhatsApp not ready" });
  }

  try {
    const chats = await client.getChats();
    res.json(
      chats.slice(0, 50).map((chat) => ({
        id: chat.id._serialized,
        name: chat.name,
        isGroup: chat.isGroup,
        unreadCount: chat.unreadCount,
        timestamp: chat.timestamp,
      }))
    );
  } catch (err) {
    console.error("Chats error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Logout endpoint
app.post("/logout", async (req, res) => {
  try {
    await client.logout();
    res.json({ success: true });
  } catch (err) {
    console.error("Logout error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// Start server and WhatsApp client
server.listen(PORT, () => {
  console.log(`WhatsApp bridge running on port ${PORT}`);
  console.log("Initializing WhatsApp client...");
  client.initialize();
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("Shutting down...");
  await client.destroy();
  server.close();
  process.exit(0);
});
