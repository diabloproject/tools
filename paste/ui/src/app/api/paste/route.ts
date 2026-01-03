import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const PASTES_DIR = path.join(process.cwd(), 'pastes');

async function ensurePastesDir() {
  try {
    await fs.access(PASTES_DIR);
  } catch {
    await fs.mkdir(PASTES_DIR, { recursive: true });
  }
}

function generateCode(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let code = '';
  for (let i = 0; i < 8; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

async function listRecentPastes() {
  await ensurePastesDir();
  
  const files = await fs.readdir(PASTES_DIR);
  const txtFiles = files.filter(f => f.endsWith('.txt'));
  
  const pasteInfos = await Promise.all(
    txtFiles.map(async (file) => {
      const filePath = path.join(PASTES_DIR, file);
      const stats = await fs.stat(filePath);
      const code = file.replace('.txt', '');
      
      // Read first 100 chars as preview
      const content = await fs.readFile(filePath, 'utf-8');
      const preview = content.substring(0, 100);
      
      return {
        code,
        createdAt: stats.mtime.toISOString(),
        preview: preview.length < content.length ? preview + '...' : preview,
      };
    })
  );
  
  // Sort by date, newest first
  return pasteInfos.sort((a, b) => 
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
}

export async function POST(request: Request) {
  const { text } = await request.json();
  
  if (!text || typeof text !== 'string') {
    return NextResponse.json({ error: 'Text is required' }, { status: 400 });
  }
  
  await ensurePastesDir();
  
  let code = generateCode();
  let filePath = path.join(PASTES_DIR, `${code}.txt`);
  
  // Ensure unique code
  while (true) {
    try {
      await fs.access(filePath);
      code = generateCode();
      filePath = path.join(PASTES_DIR, `${code}.txt`);
    } catch {
      break;
    }
  }
  
  await fs.writeFile(filePath, text, 'utf-8');
  
  return NextResponse.json({ code });
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const code = searchParams.get('code');
  
  // If no code provided, return list of recent pastes
  if (!code) {
    const pastes = await listRecentPastes();
    return NextResponse.json({ pastes });
  }
  
  await ensurePastesDir();
  const filePath = path.join(PASTES_DIR, `${code}.txt`);
  
  try {
    const text = await fs.readFile(filePath, 'utf-8');
    return NextResponse.json({ text });
  } catch {
    return NextResponse.json({ error: 'Paste not found' }, { status: 404 });
  }
}
